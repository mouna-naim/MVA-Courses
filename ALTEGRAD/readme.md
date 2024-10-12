## RNN (Reccurent Neural Network)
Un RNN traite une séquence d'entrée en générant une **représentation cachée** à chaque étape temporelle *t*, appelée, $h_{t}$. La sortie à chaque étape dépend de $x_{t}$ et l'état précédent caché $h_{t-1}$.

- À chaque étape, l'entrée $x_{t}$ est combinée avec l'état caché précédent $h_{t-1}$ via des poids (représentés par $\mathbf{U}$, $\mathbf{W}$, etc.) 

- Cet état caché est récurrent, c'est-à-dire qu'il est utilisé pour informer l'étape suivante dans la séquence.

- Problème principal des RNN classiques : Les RNNs ont des difficultés à gérer des séquences très longues, car les gradients qui se propagent au cours de plusieurs étapes peuvent "disparaître" ou exploser, ce qui entraîne la perte de l'information contextuelle à long terme. C'est là que les unités GRU et LSTM entrent en jeu pour pallier ce problème.

## GRU (Gated Recurrent Unit)
Le GRU est une amélioration des RNN classiques qui introduit des gates (portes) pour mieux gérer le flux d'informations à travers les différentes étapes temporelles. Ces gates permettent au modèle de décider quelle information doit être conservée ou oubliée.

Les GRU a deux portes principales: 
- Reset Gate $r_{t}$: Elle contrôle combien d'informations de l'état précédent $h_{t-1}$ doivent être oubliées. 
    - Si $r_{t}$ est proche de 0, le modèle oublie presque tout ce qui provient de $h_{t-1}$. Si $r_{t}$ est proche de 1, il conserve beacoup de cette informaion.
- Update gate $z_{t}$: 
    - Elle contrôle combien de l'information précédente $h_{t-1}$ sera conservée pour l'étape suivante. 
    - Si $z_{t}$ est proche de 1, l'état précédent est largement conservé. Si $z_{t}$ est proche de 0, l'état est mis à jour avec beaucoup de nouvelles informations de l'entrée actuelle.

## Exemple
Imaginez que vous ayez une séquence de mots représentant une critique de film :

"The movie was good, but the ending was terrible."
Dans un RNN classique, le modèle peut avoir du mal à réduire l'impact du mot "good" parce qu'il a été observé plus tôt dans la séquence, et cela peut fausser la prédiction globale (positive malgré la fin négative).

Avec un GRU, la porte de réinitialisation peut apprendre à ignorer "good" (puisque c'est une information ancienne) et la porte de mise à jour peut concentrer le modèle sur "terrible", ce qui influence mieux la décision finale (une classification négative).

## Self attention 
