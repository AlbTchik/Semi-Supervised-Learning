**CIFAR-10, l'apprentissage semi-supervisé et l'algorithme FixMatch**

# L'ensemble de données CIFAR-10

CIFAR-10 est un ensemble de données utilisé pour la reconnaissance
d'objets. Il s'agit d'un sous-ensemble étiqueté de l'ensemble de données
Tiny Images qui contient 10 classes d'objets: avions, voitures, oiseaux,
chats, cerfs, chiens, grenouilles, chevaux, navires et camions. Comme on
peut le voir dans la figure ci-dessous.

![CIFAR-10](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/176ae30d-8072-442f-86fa-52d82826f605)

Chaque classe contient 6 000 images en couleur de 32x32 pixels dans
l'ensemble d'apprentissage et 1 000 images dans l'ensemble de test, pour
un total de 60 000 images. Dans l'algorithme d'apprentissage
semi-supervisé seule une portion des étiquettes sera fournis à
l'algorithme.

# L'apprentissage semi-supervisé

L'apprentissage semi-supervisé utilise une combinaison de données
étiquetées et non étiquetées pour entraîner un modèle prédictif.
Celui-ci doit donc découvrir la distribution sous-jacente des données
non étiquetés. Il dispose aussi de certaines données étiquetée pour
aiguiller la recherche en fixant une représentation initial de chacune
des classes.


![supervised learning](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/c06a608f-8138-43b0-a336-d1170bb29134)

L'avantage de cette technique d'apprentissage est que les données non
étiquetées sont souvent moins coûteuses à obtenir et plus abondantes que
les données étiquetées. Cependant, toute la difficulté de
l'apprentissage semi-supervisé réside dans la façon d'utiliser au mieux
les données non étiquetés pour l'apprentissage.

## L'algorithme MixMatch

En 2019, Berthelot et al. [@berthelot2019mixmatch] proposent \"MixMatch:
A Holistic Approach to Semi-Supervised Learning\". Cette technique a
permis d'ouvrir de nouvelles pistes pour l'apprentissage semi-supervisé.

Dans MixMatch, les données étiquetés sont traité de manière supervisé,
comme dans un classifieur classique. Une augmentation de données est
appliqué à chaque donnée non étiquetée et le modèle réalise une
inférence sur K versions de la même image. Puis les prédictions des
images annotés sont moyennées pour obtenir la prédiction finale, comme
on peut le voir sur la figure ci-dessous.

![mixmatch](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/94f512d4-5450-4ec8-916d-c8ecb328a907)

Le modèle est mis à jour en utilisant une régularisation qui combine une
entropie croisée pour les exemples supervisés et une entropie croisée
pour les exemples non supervisés.\
La même année, Berthelot et al. [@berthelot2019remixmatch] publient une
amélioration de l'algorithme MixMatch, appelée \"Remixmatch:
Semi-supervised learning with distribution alignment and augmentation
anchoring\" qui introduit un réseau de neurones à attention dans le
processus d'auto-étiquetage et de mélange des données. L'utilisation de
ce réseau permet de mieux identifier les parties importantes des données
non étiquetées et de les utiliser de manière plus efficace pour
améliorer la performance du modèle.

# L'algorithme FixMatch

En 2020, Sohn et al. [@sohn2020fixmatch] publient l'article \"Fixmatch:
Simplifying semi-supervised learning with consistency and confidence\"
qui propose une nouvelle méthode d'apprentissage semi-supervisé
similaire à MixMatch.

## Augmentation des données

Cet algorithme utilise aussi une augmentation aléatoire des images non
étiquetées, similaire à MixMatch. Cependant, FixMatch exploite deux
types d'augmentations : les augmentations \"faibles\" et \"fortes\".\
L'augmentation faible met en place des retournements et des décalages.
Plus précisément, les images ont une probabilité de 50% d'être retournée
et 12,5% d'être retournées verticalement et horizontalement. Pour
l'augmentation forte, la méthode RandAugment basée sur l'article de
Cubuk et al.[@cubuk2020randaugment] a été employée. Celle-ci met en
place un apprentissage par renforcement pour trouver une stratégie
d'augmentation adaptée avec des transformations d'images basiques.
Certaines variations de l'image sont donc fortement augmentés tandis que
d'autres sont très proches de l'image originale.

## Régularisation par cohérence

L'avantage principal de FixMatch est de contraindre le modèle à prédire
les mêmes classes pour chacune des variations d'une image non étiquetée,
comme on peut le voir dans la figure ci-dessous.

![coherence loss](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/6169d026-6655-4df1-81dd-2bb74e3eda5b)

Pour cela, on défini une pseudo-étiquette. Celle-ci représente la
prédiction de l'image la moins augmentée et sert de référence.
L'objectif est que toutes les variations de cette image prédisent la
classe de la pseudo-étiquette. Ce principe est mis en place via la
régularisation par cohérence, qui est intégrée dans la perte non
supervisée, présentée dans la formule ci-dessous :

$$L_{u} = \frac{1}{\mu B}\sum ^{\mu B} _{b=1} {(max(q_{b}) \geq \tau)} H( \hat{q}_{b} , p_{m}(y | A(u_{b})))$$
Avec $H( )$ l'entropie croisée, $q_{b} = p_{m}(y|\alpha(u_{b}))$ la
probabilité des classes, $\hat{q}_{b} = arg max(q_{b})$ la
pseudo-étiquette et $\tau$ le seuil permettant de choisir la
pseudo-étiquette.\
Cette formule permet de tirer pleinement partie des perturbations
aléatoire pour améliorer la généralisation du modèle. La formule de la
perte supervisée est :

$$L_{s} = \frac{1}{B} \sum ^{B} _{b=1} {H( p_{b}, p_{m} ( y | \alpha(x_{b})))}$$

Finalement, à la sortie du réseau, la fonction de perte de FixMatch est
calculé en combinant une perte non supervisé et une perte supervisée, de
la même manière que MixMatch. La perte totale du modèle est donc :

$$L_{FixMatch} = L_{u} +\lambda_{u} L_{s}$$ Avec $\lambda_{u}$ un
scalaire représentant l'importance de la perte non supervisée.

## Le modèle Wide ResNet

FixMatch fait intervenir un classifieur qui sert de base à
l'entraînement. Le modèle Wide ResNet de Zagoruyko et
al.[@zagoruyko2016wide] est une version élargit du modèle ResNet, qui
intègre aussi un dropout dans les connections résiduelles. L'avantage du
modèle ResNet, crée par He et al.[@he2016deep] en 2015, est l'ajout de
connections résiduelles qui permet d'obtenir des réseaux d'une
profondeur inégalée à l'époque. Celles-ci permettent aux réseau
d'apprendre au mieux les transformations en comparant le résultat d'une
opération avec l'entrée originelle.

### Les problèmes du ResNet

Bengio and al.[@bengio2007scaling] montrent que les circuits peu
profonds peuvent nécessiter exponentiellement plus de composants que les
circuits plus profonds. C'est pour cela que les auteurs du ResNet ont
essayé de le rendre aussi mince que possible afin d'augmenter sa
profondeur et d'avoir moins de paramètres. Ils ont même introduit un
bloc \"goulot d'étranglement\" qui rend les blocs du ResNet encore plus
minces.

Cependant, lorsque le gradient circule dans le réseau, rien ne l'oblige
à passer par les poids résiduels des blocs. Ainsi, seuls quelques blocs
apprennent des représentations utiles ou de nombreux blocs partagent
très peu d'informations et ne contribuent que faiblement à l'objectif
final.

### l'avantage du Wide ResNet

Le bloc résiduel du Wide ResNet possède l'architecture indiqué dans la
partie (d) de la figure ci-dessous.

![wide resnet](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/d2a462b6-59a1-4ddf-b06f-91c934345cdb)

On voit que la connection résiduelle est beaucoup plus large que pour un
ResNet classique (partie (a) et (b) de la figure). En élargissant la
couche résiduelle, on obtient un réseau moins profond avec une précision
équivalente. De plus, l'ajout du dropout dans les connections
résiduelles permet de réduire la dégradation des poids.

Le Wide ResNet a permis de démontrer que la puissance principale des
réseaux résiduels réside dans les blocs résiduels, et non dans la
profondeur extrême. Un réseau moins profond signifie donc un nombre de
couches réduit et un temps d'entraînement plus court, pour une
performance équivalente.

# Mise en pratique de FixMatch

## Implémentation de l'algorithme

Le choix a été fait de ne pas réimplementer FixMatch depuis le début.
Ainsi, le répertoire de
[kekmodel](#https://github.com/kekmodel/FixMatch-pytorch) a été utilisé
comme base pour ce projet. Il faut évidemment remercier Jungdae Kim pour
avoir déposer son implémentation du FixMatch sur GitHub.\
Cette implémentation de FixMatch apporte quelques améliorations pratique
pour l'entraînement des modèles. Par exemple, le Wide ResNet utilisé
comprend 3 bloc résiduels, contenant chacun 6 couches de convolution qui
élargissent le nombre de channel pour passer de 16 à 128, en doublant à
chaque fois. L'optimiseur utilisée est AdamW, une variante d'Adam dans
laquelle la décroissance du poids n'est effectuée qu'après avoir
contrôlé la taille du pas en fonction des paramètres. Aussi, le taux
d'apprentissage suit une évolution sous la forme d'un cosinus qui perd
en amplitude au fur à mesure des epochs.

## Entraînement du modèle

Malheureusement, du fait du long entraînement de l'algorithme (1
journée), de la puissance de calcul limitée, des considérations
écologiques, et aussi car la plupart des expérimentations ont déjà été
réalisée dans l'étude d'ablation de l'article de Sohn et
al.[@sohn2020fixmatch], il as été possible de lancer que quelques
itérations de l'entraînement.\
Dans un premier temps, un entraînement du programme sans modification a
été réalisé. Les hyperparamètres choisis sont les suivants : Taux
d'apprentissage de 0.03, Réduction du taux d'apprentissage jusqu'à 5e-4,
Seuil des pseudo-étiquettes de 0.95, Taille d'échantillon de 64, 250
images annotés sur 60 000 et 400 epochs. L'évolution de des accuracy et
des pertes pendant l'entraînement sont présentés dans la figure
ci-dessous.

![FixMatch acc](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/5f2f8b58-22fb-4c58-a1b4-55ee8505362a)</p>

![fixmatch losses](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/eee9027b-642f-40c7-8162-c48c4b22efc8)

On voit que l'entraînement est un succès et qu'il mène à des résultats
proche de l'état de l'art. La perte met du temps à descendre au début et
cela est dû au petit nombre de données étiquetés.\
Pour l'expérimentation suivante, il a été décidé de vérifier l'intérêt
de la réduction du taux d'apprentissage (scheduler) sous forme d'un
cosinus. Il a donc été retiré et l'entraînement a été ensuite relancé
pour 200 epochs. La comparaison des résultats est présenté dans la
figure ci-dessous.

![acc comparaison](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/75e71aba-9d20-44f7-a335-bbb50761da2e)</p>

![loss comparaison](https://github.com/AlbTchik/Semi-Supervised-Learning/assets/90097422/ef0ed5b6-34e6-4c9a-ab05-2cc333d2155d)

On voit que la réduction du taux d'apprentissage sous la forme d'un
cosinus qui perd en amplitude est utile pour améliorer les performances
du modèle. Surtout au début de l'entraineement, où la perte augmente
fortement pendant les 40 premières epochs. Ce phénomène est dû a une
erreur dans les pseudo étiquettes, ce qui force le modèle à converger
vers de mauvais résultats. Ce phénomène est beaucoup plus important
quand le scheduler n'est pas présent car le taux d'apprentissage est
assez élevé, et va rapidement dans la mauvaise direction.\
L'intérêt principal du scheduler est d'accélérer la convergence du
modèle, ce qui permet donc d'obtenir de meilleurs résultats en un temps
équivalent, comme on peut le voir sur la figure.\
Au terme de chacun des entrainements, on obtient donc les résultats
suivants :

<figure>
<table>
<thead>
<tr class="header">
<th style="text-align: left;">Modèles</th>
<th style="text-align: left;">Top-5 Accuracy</th>
<th style="text-align: left;">Top-1 Accuracy</th>
<th style="text-align: left;"></th>
<th style="text-align: left;"></th>
<th style="text-align: left;"></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">FixMatch avec Scheduler</td>
<td style="text-align: left;"><span
class="math inline">99.82</span></td>
<td style="text-align: left;"><span class="math inline">92.0</span></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;">FixMatch sans Scheduler</td>
<td style="text-align: left;"><span
class="math inline">99.46</span></td>
<td style="text-align: left;"><span
class="math inline">88.51</span></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
</tbody>
</table>
<figcaption>Comparaison des modèles</figcaption>
</figure>

Comme attendu, le modèle avec la routine de réduction du taux
d'apprentissage obtient de meilleurs performances. Pour que cette étude
soit réellement significative, il aurait fallu répéter plusieurs fois
l'expérience. Il aurait aussi été possible d'analyser d'autres éléments
utilisés par cette implémentation de FixMatch, comme un changement de
réseau, de méthode d'augmentation, etc\... On aurait pu aussi comparer
les résultats de FixMatch avec ceux de MixMatch ou ReMixMatch.
