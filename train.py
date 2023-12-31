import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import csv

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import get_cifar10
from utils import AverageMeter, accuracy
from models.ema import ModelEMA

logger = logging.getLogger(__name__)
best_acc = 0

def create_model(args):
    import models.wideresnet as models
    model = models.build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=10)
    logger.info("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
    return model

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--total-steps', default=200*200, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=200, type=int, help='number of eval steps to run')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--out', default='result', help='directory to output the result')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    args = parser.parse_args()

    global best_acc
    
    #Set Cuda device and print infos
    device = torch.device('cuda', 0)
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.warning(f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ",)
    logger.info(dict(args._get_kwargs()))

    #Set seed
    if args.seed is not None:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    #Load data
    labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, './data')
    train_sampler = RandomSampler
    labeled_trainloader = DataLoader(labeled_dataset, sampler=train_sampler(labeled_dataset), batch_size=args.batch_size, num_workers=4, drop_last=True)
    unlabeled_trainloader = DataLoader(unlabeled_dataset, sampler=train_sampler(unlabeled_dataset), batch_size=args.batch_size*args.mu, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size, num_workers=4)

    #create model
    model = create_model(args)
    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any( nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    #optimizer, scheduler and EMA
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=True)
    #scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.total_steps)

    ema_model = ModelEMA(args, model, args.ema_decay)
    
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    args.start_epoch = 0

    #load checkpoint if any
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])

    #Start the training
    logger.info("***** Start training *****")
    logger.info(f"  Task = cifar10@250")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    test_accs = []
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    #training mode
    model.train()

    metrics = []

    #for each epochs
    for epoch in range(args.start_epoch, args.epochs):

        #Set tenser that can easily compute averages
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        #set tqdm training progression bar
        p_bar = tqdm(range(args.eval_step), disable=False)

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            loss = Lx + 1 * Lu

            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            #scheduler.step()
            ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

            #tqdm training progression bar print
            #p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(epoch=epoch + 1, epochs=args.epochs, batch=batch_idx + 1, iter=args.eval_step, lr=scheduler.get_last_lr()[0], data=data_time.avg, bt=batch_time.avg, loss=losses.avg, loss_x=losses_x.avg, loss_u=losses_u.avg, mask=mask_probs.avg))
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(epoch=epoch + 1, epochs=args.epochs, batch=batch_idx + 1, iter=args.eval_step, lr=args.lr, data=data_time.avg, bt=batch_time.avg, loss=losses.avg, loss_x=losses_x.avg, loss_u=losses_u.avg, mask=mask_probs.avg))
            p_bar.update()

        p_bar.close()

        #test the model
        test_model = ema_model.ema
        test_loss, test_acc, test_top_5 = test(args, test_loader, test_model, epoch)

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

        metrics.append([epoch, losses.avg, losses_x.avg, losses_u.avg, test_acc, test_top_5, test_loss])

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        model_to_save = model.module if hasattr(model, "module") else model
        ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema
        #save_checkpoint({'epoch': epoch + 1, 'state_dict': model_to_save.state_dict(), 'ema_state_dict': ema_to_save.state_dict(), 'acc': test_acc, 'best_acc': best_acc, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), }, is_best, args.out)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model_to_save.state_dict(), 'ema_state_dict': ema_to_save.state_dict(), 'acc': test_acc, 'best_acc': best_acc, 'optimizer': optimizer.state_dict(), }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

    args.writer.close()

    with open('./computerVision_projet2/FixMatch-pytorch/results/normal_run/run2.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics)


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    #tqdm training progression bar
    test_loader = tqdm(test_loader, disable=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            #tqdm training progression bar print
            test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(batch=batch_idx + 1, iter=len(test_loader), data=data_time.avg, bt=batch_time.avg, loss=losses.avg, top1=top1.avg, top5=top5.avg,))
        test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
