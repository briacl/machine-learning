# Explication Technique de MyChatGPT

Ce document détaille le fonctionnement interne du script `mychatgpt.py`. Ce programme est un simulateur pédagogique de LLM (Large Language Model) "from scratch", sans utiliser de librairies de machine learning (comme PyTorch ou TensorFlow), pour comprendre la mécanique sous-jacente.

## 1. Concepts Mathématiques de Base

Les fonctions situées au début du script sont le moteur mathématique du réseau de neurones.

### A. Sigmoïde (`sigmoide`)
$$ f(x) = \frac{1}{1 + e^{-x}} $$
*   **Rôle** : Fonction d'activation. Elle transforme n'importe quel nombre en une valeur comprise entre 0 et 1.
*   **Usage dans le code** : Utilisée dans la **couche cachée**. Elle décide si un neurone est "actif" ou non en fonction des mots qu'il a vus en entrée.

### B. Softmax (`softmax`)
$$ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum e^{z_j}} $$
*   **Rôle** : Transforme une liste de scores bruts (logits) en **probabilités**. La somme des sorties vaut toujours 1 (100%).
*   **Usage dans le code** : Utilisée à la sortie du réseau pour que le modèle nous dise "Je suis sûr à 80% que le prochain mot est 'bonjour'".

---

## 2. Le Vocabulaire et la Tokenization

### Tokenization (`tokenize`)
Les ordinateurs ne lisent pas le texte, ils lisent des nombres.
*   Le script prend une phrase brute : `"Bonjour, ça va?"`
*   Il sépare la ponctuation : `"Bonjour , ça va ?"`
*   Il découpe en liste (tokens) : `['bonjour', ',', 'ça', 'va', '?']`

### Classe `Vocabulary`
Elle maintient un dictionnaire bidirectionnel :
*   **Mot vers Index** : `chat` -> `42` (pour l'entrée du réseau)
*   **Index vers Mot** : `42` -> `chat` (pour décoder la sortie)

---

## 3. Le Cerveau : `class MyChatGPT`

C'est un **Réseau de Neurones à Propagation Avant (Feedforward Neural Network)** simple.

### Architecture
1.  **Couche d'Entrée (Input Layer)** : Taille = Taille du vocabulaire.
    *   Représentation **Bag-of-Words** : Si le mot "chat" est présent dans le contexte, l'entrée correspondant à l'index 42 est mise à 1.0.
2.  **Couche Cachée (Hidden Layer)** : Taille configurable (par défaut 64 neurones).
    *   C'est la "mémoire de travail" ou "l'intelligence" du modèle. Chaque neurone apprend à détecter des motifs spécifiques (ex: un neurone pourrait s'activer fortement s'il voit "comment" ET "allez").
3.  **Couche de Sortie (Output Layer)** : Taille = Taille du vocabulaire.
    *   Chaque neurone de sortie correspond à un mot candidat pour la suite.

### Les Matrices de Poids (Synapses)
*   `self.W1` : Connexions entre **Entrée** et **Caché**.
*   `self.W2` : Connexions entre **Caché** et **Sortie**.
*   Ces matrices contiennent des nombres aléatoires au début, et sont ajustées (apprises) au fil du temps.

---

## 4. La Prédiction (`predict_next_word`)

C'est la "Passe Avant" (Forward Pass).
1.  **Encodage** : On transforme les mots du contexte en vecteur d'entrée $X$.
2.  **Calcul Caché** : $H = \text{sigmoide}(X \cdot W1)$. Les neurones cachés s'activent.
3.  **Calcul Sortie** : $Z = H \cdot W2$. On obtient les logits pour chaque mot possible.
4.  **Probabilités** : $P = \text{softmax}(Z)$. On obtient la distribution de probabilité sur tout le vocabulaire.

---

## 5. La Génération (`generate_response`)

Les LLMs ne génèrent pas des phrases, ils génèrent **un mot à la suite de l'autre**.
1.  Prendre le texte utilisateur (prompt).
2.  Prédire le prochain mot avec `predict_next_word`.
3.  Choisir le mot avec la plus haute probabilité (**Greedy Decoding**).
4.  Ajouter ce mot au contexte.
5.  Répéter jusqu'à un point final, une répétition, ou la longueur max.

---

## 6. L'Apprentissage (`learn`)

C'est ici que la magie opère. L'algorithme utilisé est la **Descente de Gradient (Gradient Descent)** via **Rétropropagation (Backpropagation)**.

Quand vous dites au modèle "Tu aurais dû dire 'Bonjour'" :
1.  **Calcul de l'Erreur** : Différence entre ce qu'il a prédit et 1.0 (la certitude absolue pour le bon mot).
2.  **Mise à jour W2** : On modifie les poids reliant la couche cachée à la sortie.
    *   Si le neurone caché était actif et qu'il fallait dire "Bonjour", on **renforce** le lien.
3.  **Rétropropagation de l'erreur** : On calcule la "part de responsabilité" de chaque neurone caché dans l'erreur finale.
4.  **Mise à jour W1** : On modifie les poids reliant l'entrée à la couche cachée en fonction de cette responsabilité.

C'est ce processus, répété des centaines de fois (`epochs`), qui "grave" l'information dans les matrices de poids.

---

## 7. Persistance

*   Le cerveau (les matrices `W1`, `W2` et la taille `hidden_size`) est sauvegardé dans `chat_brain_v2.json`.
*   Cela permet au modèle de ne pas redevenir "stupide" à chaque redémarrage.
