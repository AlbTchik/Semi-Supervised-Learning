\documentclass{article}
\usepackage[a4paper, total={6.5in, 9in}]{geometry}
\usepackage[utf8]{inputenc}

 \usepackage{graphicx}
 \usepackage{subcaption}
 \usepackage{float}
 
\usepackage{natbib}

\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    }
\urlstyle{same}




\title{\textbf{CIFAR-10, l'apprentissage semi-supervisé et l'algorithme FixMatch}}
\author{Alban Tchikladzé}
\date{Avril 2023}

\begin{document}

\maketitle
\tableofcontents
\newpage

\section{L'ensemble de données CIFAR-10}

CIFAR-10 est un ensemble de données utilisé pour la reconnaissance d'objets. Il s'agit d'un sous-ensemble étiqueté de l'ensemble de données Tiny Images qui contient 10 classes d'objets: avions, voitures, oiseaux, chats, cerfs, chiens, grenouilles, chevaux, navires et camions. Comme on peut le voir dans la figure ci-dessous.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.3]{Images/CIFAR-10}
    \caption{Exemples d'images de CIFAR-10}
\end{figure}

Chaque classe contient 6 000 images en couleur de 32x32 pixels dans l'ensemble d'apprentissage et 1 000 images dans l'ensemble de test, pour un total de 60 000 images. Dans l'algorithme d'apprentissage semi-supervisé seule une portion des étiquettes sera fournis à l'algorithme.

\section{L'apprentissage semi-supervisé}

L'apprentissage semi-supervisé utilise une combinaison de données étiquetées et non étiquetées pour entraîner un modèle prédictif. Celui-ci doit donc découvrir la distribution sous-jacente des données non étiquetés. Il dispose aussi de certaines données étiquetée pour aiguiller la recherche en fixant une représentation initial de chacune des classes.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.15]{Images/supervised learning}
    \caption{Fonctionnement de l'apprentissage semi-supervisé}
\end{figure}

L'avantage de cette technique d'apprentissage est que les données non étiquetées sont souvent moins coûteuses à obtenir et plus abondantes que les données étiquetées. Cependant, toute la difficulté de l'apprentissage semi-supervisé réside dans la façon d'utiliser au mieux les données non étiquetés pour l'apprentissage. 

\subsection{L'algorithme MixMatch}

En 2019, Berthelot et al. \cite{berthelot2019mixmatch} proposent "MixMatch: A Holistic Approach to Semi-Supervised Learning". Cette technique a permis d'ouvrir de nouvelles pistes pour l'apprentissage semi-supervisé.

Dans MixMatch, les données étiquetés sont traité de manière supervisé, comme dans un classifieur classique. Une augmentation de données est appliqué à chaque donnée non étiquetée et le modèle réalise une inférence sur K versions de la même image. Puis les prédictions des images annotés sont moyennées pour obtenir la prédiction finale, comme on peut le voir sur la figure ci-dessous.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.25]{Images/mixmatch}
    \caption{Fonctionnement de l'algorithme MixMatch}
\end{figure}

Le modèle est mis à jour en utilisant une régularisation qui combine une entropie croisée pour les exemples supervisés et une entropie croisée pour les exemples non supervisés.
\\

La même année, Berthelot et al. \cite{berthelot2019remixmatch} publient une amélioration de l'algorithme MixMatch, appelée  "Remixmatch: Semi-supervised learning with distribution alignment and augmentation anchoring" qui introduit un réseau de neurones à attention dans le processus d'auto-étiquetage et de mélange des données. L'utilisation de ce réseau permet de mieux identifier les parties importantes des données non étiquetées et de les utiliser de manière plus efficace pour améliorer la performance du modèle. 

\section{L'algorithme FixMatch}

En 2020, Sohn et al. \cite{sohn2020fixmatch} publient l'article "Fixmatch: Simplifying semi-supervised learning with consistency and confidence" qui propose une nouvelle méthode d'apprentissage semi-supervisé similaire à MixMatch.

\subsection{Augmentation des données}

 Cet algorithme utilise aussi une augmentation aléatoire des images non étiquetées, similaire à MixMatch. Cependant, FixMatch exploite deux types d'augmentations : les augmentations "faibles" et "fortes". 
 \\
 L'augmentation faible met en place des retournements et des décalages. Plus précisément, les images ont une probabilité de 50\% d'être retournée et 12,5\% d'être retournées verticalement et horizontalement. Pour l'augmentation forte, la méthode RandAugment basée sur l'article de Cubuk et al.\cite{cubuk2020randaugment} a été employée. Celle-ci met en place un apprentissage par renforcement pour trouver une stratégie d'augmentation adaptée avec des transformations d'images basiques. Certaines variations de l'image sont donc fortement augmentés tandis que d'autres sont très proches de l'image originale.

 \subsection{Régularisation par cohérence}

 L'avantage principal de FixMatch est de contraindre le modèle à prédire les mêmes classes pour chacune des variations d'une image non étiquetée, comme on peut le voir dans la figure ci-dessous.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.8]{Images/coherence loss}
    \caption{Régularisation par cohérence}
\end{figure}

Pour cela, on défini une pseudo-étiquette. Celle-ci représente la prédiction de l'image la moins augmentée et sert de référence. L'objectif est que toutes les variations de cette image prédisent la classe de la pseudo-étiquette. Ce principe est mis en place via la régularisation par cohérence, qui est intégrée dans la perte non supervisée, présentée dans la formule ci-dessous :

\begin{equation}
    L_{u} = \frac{1}{\mu B}\sum ^{\mu B} _{b=1} {(max(q_{b}) \geq \tau)} H( \hat{q}_{b} , p_{m}(y | A(u_{b})))
\end{equation}
Avec $H( )$ l'entropie croisée, $q_{b} = p_{m}(y|\alpha(u_{b}))$ la probabilité des classes, $\hat{q}_{b} = arg max(q_{b})$ la pseudo-étiquette et $\tau$ le seuil permettant de choisir la pseudo-étiquette.   
\\

Cette formule permet de tirer pleinement partie des perturbations aléatoire pour améliorer la généralisation du modèle. La formule de la perte supervisée est : 

\begin{equation}
    L_{s} = \frac{1}{B} \sum ^{B} _{b=1} {H( p_{b}, p_{m} ( y | \alpha(x_{b})))}
\end{equation}

Finalement, à la sortie du réseau, la fonction de perte de FixMatch est calculé en combinant une perte non supervisé et une perte supervisée, de la même manière que MixMatch. La perte totale du modèle est donc :

\begin{equation}
    L_{FixMatch} = L_{u} +\lambda_{u} L_{s}
\end{equation}
Avec $\lambda_{u}$ un scalaire représentant l'importance de la perte non supervisée.

\subsection{Le modèle Wide ResNet}

FixMatch fait intervenir un classifieur qui sert de base à l'entraînement. Le modèle Wide ResNet de Zagoruyko et al.\cite{zagoruyko2016wide} est une version élargit du modèle ResNet, qui intègre aussi un dropout dans les connections résiduelles. L'avantage du modèle ResNet, crée par He et al.\cite{he2016deep} en 2015, est l'ajout de connections résiduelles qui permet d'obtenir des réseaux d'une profondeur inégalée à l'époque. Celles-ci permettent aux réseau d'apprendre au mieux les transformations en comparant le résultat d'une opération avec l'entrée originelle.

\subsubsection{Les problèmes du ResNet}

Bengio and al.\cite{bengio2007scaling} montrent que les circuits peu profonds peuvent nécessiter exponentiellement plus de composants que les circuits plus profonds. C'est pour cela que les auteurs du ResNet ont essayé de le rendre aussi mince que possible afin d'augmenter sa profondeur et d'avoir moins de paramètres. Ils ont même introduit un bloc "goulot d'étranglement" qui rend les blocs du ResNet encore plus minces. 

Cependant, lorsque le gradient circule dans le réseau, rien ne l'oblige à passer par les poids résiduels des blocs. Ainsi, seuls quelques blocs apprennent des représentations utiles ou de nombreux blocs partagent très peu d'informations et ne contribuent que faiblement à l'objectif final.

\subsubsection{l'avantage du Wide ResNet}

Le bloc résiduel du Wide ResNet possède l'architecture indiqué dans la partie (d) de la figure ci-dessous.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.25]{Images/wide resnet}
    \caption{Les différents bloc residuels}
\end{figure}

On voit que la connection résiduelle est beaucoup plus large que pour un ResNet classique (partie (a) et (b) de la figure). En élargissant la couche résiduelle, on obtient un réseau moins profond avec une précision équivalente. De plus, l'ajout du dropout dans les connections résiduelles permet de réduire la dégradation des poids.

Le Wide ResNet a permis de démontrer que la puissance principale des réseaux résiduels réside dans les blocs résiduels, et non dans la profondeur extrême. Un réseau moins profond signifie donc un nombre de couches réduit et un temps d'entraînement plus court, pour une performance équivalente.

\section{Mise en pratique de FixMatch}

\subsection{Implémentation de l'algorithme}

Le choix a été fait de ne pas réimplementer FixMatch depuis le début. Ainsi, le répertoire de \hyperlink{https://github.com/kekmodel/FixMatch-pytorch}{kekmodel} a été utilisé comme base pour ce projet. Il faut évidemment remercier Jungdae Kim pour avoir déposer son implémentation du FixMatch sur GitHub.
\\

Cette implémentation de FixMatch apporte quelques améliorations pratique pour l'entraînement des modèles. Par exemple, le Wide ResNet utilisé comprend 3 bloc résiduels, contenant chacun 6 couches de convolution qui élargissent le nombre de channel pour passer de 16 à 128, en doublant à chaque fois. L'optimiseur utilisée est AdamW, une variante d'Adam dans laquelle la décroissance du poids n'est effectuée qu'après avoir contrôlé la taille du pas en fonction des paramètres. Aussi, le taux d'apprentissage suit une évolution sous la forme d'un cosinus qui perd en amplitude au fur à mesure des epochs.

\subsection{Entraînement du modèle}

Malheureusement, du fait du long entraînement de l'algorithme (1 journée), de la puissance de calcul limitée, des considérations écologiques, et aussi car la plupart des expérimentations ont déjà été réalisée dans l'étude d'ablation de l'article de Sohn et al.\cite{sohn2020fixmatch}, il as été possible de lancer que quelques itérations de l'entraînement. 
\\

Dans un premier temps, un entraînement du programme sans modification a été réalisé. Les hyperparamètres choisis sont les suivants :  Taux d'apprentissage de 0.03, Réduction du taux d'apprentissage jusqu'à 5e-4, Seuil des pseudo-étiquettes de 0.95, Taille d'échantillon de 64, 250 images annotés sur 60 000 et 400 epochs. L'évolution de des accuracy et des pertes pendant l'entraînement sont présentés dans la figure ci-dessous.


\begin{figure}[H]
    \begin{subfigure}[c]{.43\linewidth}
        \centering
        \includegraphics[width=\textwidth]{Images/FixMatch acc}
        \caption{Accuracy pendant l'entrainement}
    \end{subfigure}
    \hfill%
    \begin{subfigure}[c]{.43\linewidth}
        \centering
        \includegraphics[width=\textwidth]{Images/fixmatch losses}
        \caption{Perte pendant l'entraînement}
    \end{subfigure}
    \caption{Analyse de l'entrainement de FixMatch}
\end{figure}

On voit que l'entraînement est un succès et qu'il mène à des résultats proche de l'état de l'art. La perte met du temps à descendre au début et cela est dû au petit nombre de données étiquetés.
\\

Pour l'expérimentation suivante, il a été décidé de vérifier l'intérêt de la réduction du taux d'apprentissage (scheduler) sous forme d'un cosinus. Il a donc été retiré et l'entraînement a été ensuite relancé pour 200 epochs. La comparaison des résultats est présenté dans la figure ci-dessous.

\begin{figure}[H]
    \begin{subfigure}[c]{.43\linewidth}
        \centering
        \includegraphics[width=\textwidth]{Images/acc comparaison}
        \caption{Accuracy pendant l'entrainement}
    \end{subfigure}
    \hfill%
    \begin{subfigure}[c]{.43\linewidth}
        \centering
        \includegraphics[width=\textwidth]{Images/loss comparaison}
        \caption{Pertes pendant l'entraînement}
    \end{subfigure}
    \caption{Comparaison des modèles FixMatch}
\end{figure}

On voit que la réduction du taux d'apprentissage sous la forme d'un cosinus qui perd en amplitude est utile pour améliorer les performances du modèle. Surtout au début de l'entraineement, où la perte augmente fortement pendant les 40 premières epochs. Ce phénomène est dû a une erreur dans les pseudo étiquettes, ce qui force le modèle à converger vers de mauvais résultats. Ce phénomène est beaucoup plus important quand le scheduler n'est pas présent car le taux d'apprentissage est assez élevé, et va rapidement dans la mauvaise direction.
\\

L'intérêt principal du scheduler est d'accélérer la convergence du modèle, ce qui permet donc d'obtenir de meilleurs résultats en un temps équivalent, comme on peut le voir sur la figure.
\\

Au terme de chacun des entrainements, on obtient donc les résultats suivants : 

\renewcommand{\arraystretch}{1.5} 
\begin{figure}[H]
    \centering
    \begin{tabular}{ | l | l | l | l | l | l | }
     \hline			
       Modèles & Top-5 Accuracy & Top-1 Accuracy \\ \hline \hline
       FixMatch avec Scheduler & $99.82$ & $92.0$ \\ \hline
       FixMatch sans Scheduler & $99.46$ & $88.51$ \\
     \hline  
     \end{tabular}
     \caption{Comparaison des modèles}
 \end{figure}
Comme attendu, le modèle avec la routine de réduction du taux d'apprentissage obtient de meilleurs performances. Pour que cette étude soit réellement significative, il aurait fallu répéter plusieurs fois l'expérience.
Il aurait aussi été possible d'analyser d'autres éléments utilisés par cette implémentation de FixMatch, comme un changement de réseau, de méthode d'augmentation, etc... On aurait pu aussi comparer les résultats de FixMatch avec ceux de MixMatch ou ReMixMatch.

\newpage

\bibliographystyle{unsrt}
\bibliography{Bibliographie.bib}


\end{document}
