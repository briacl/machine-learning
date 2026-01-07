# Le Perceptron : Un neurone artificiel simplifié

Ce document explique le fonctionnement du script `perceptron.py`, qui simule la prise de décision d'un neurone artificiel.

## Concept Général
Le programme simule comment un neurone prend une décision.
Imaginez que ce neurone doit décider : **"Est-ce que je vais au cinéma ?"**

## 1. Les ENTRÉES (Inputs)
Ce sont les signaux reçus (ex: informations du monde extérieur).
*   **Input 1 (`x1`)** : Est-ce qu'il fait beau ? (2.2 = très beau)
*   **Input 2 (`x2`)** : Est-ce que le film est bien noté ? (3.4 = excellente note)
*   **Input 3 (`x3`)** : Est-ce que j'ai des amis dispo ? (1.6 = quelques amis)

## 2. Les POIDS (Weights) - L'importance de chaque information
Le neurone (vous) accorde une importance différente à chaque entrée.
*   **Poids 1 (Météo)** : 2.1 (Important, j'aime le beau temps)
*   **Poids 2 (Note film)** : 1.3 (Moyennement important, je suis bon public)
*   **Poids 3 (Amis)** : 6.7 (TRÈS important, je n'aime pas y aller seul)

## 3. Le BIAIS (Bias) - La facilité à s'activer
C'est une sorte de prédisposition naturelle.
*   **Bias = 3** : Je suis de nature optimiste, j'ai envie de sortir à la base.
*   Si le biais était négatif (-3), il faudrait beaucoup d'arguments pour me convaincre.

## 4. Le CALCUL (Weighted Sum)
Le neurone fait la somme de toutes les infos pondérées par leur importance.
**Formule** : `(Info1 * Importance1) + (Info2 * Importance2) + ... + Prédisposition`

## 5. La DÉCISION (Fonction d'activation)
Si le résultat (potentiel d'activation) est très haut, le neurone "s'active" (envoie un signal).
Par exemple, un résultat de **22.76** est un signal très fort !

## Fonctionnalités du Code
*   **Visualisation** : La fonction `visualiser_perceptron` dessine l'architecture du neurone avec Matplotlib (Style Cyberpunk).
