import math
import random
import json
import os

# --- 1. Outils Mathématiques (Le "Moteur") ---

def softmax(logits):
    """Transforme les scores bruts (logits) en probabilités (%)."""
    # Pour éviter les trop grands nombres (overflow), on soustrait le max
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    sum_exps = sum(exps)
    probs = [e / sum_exps for e in exps]
    return probs

def sigmoide(x):
    """Fonction d'activation pour les neurones cachés (0 à 1)."""
    return 1 / (1 + math.exp(-x))

def derivee_sigmoide(output):
    """Dérivée pour l'apprentissage (rétropropagation)."""
    return output * (1 - output)

# --- 2. La Mémoire des Mots (Vocabulaire) ---

class SimpleVocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    
    def add(self, word):
        word = word.lower() # On travaille en minuscules pour simplifier
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
    
    def get_idx(self, word):
        return self.word2idx.get(word.lower())
    
    def get_word(self, idx):
        if 0 <= idx < len(self.idx2word):
            return self.idx2word[idx]
        return None
    
    def __len__(self):
        return len(self.idx2word)

    def to_list(self):
        return self.idx2word
    
    @staticmethod
    def from_list(word_list):
        v = SimpleVocabulary()
        for w in word_list:
            v.add(w)
        return v

# --- 3. Le Cerveau (Réseau de Neurones) ---

class PedagogicalLLM:
    def __init__(self, vocabulary, hidden_size=5):
        self.vocab = vocabulary
        self.hidden_size = hidden_size
        self.learning_rate = 0.5 # Taux d'apprentissage (vitesse à laquelle il change d'avis)

        # Initialisation aléatoire des poids
        # W1 : Poids entre Entrée (Mots du contexte) -> Neurones Cachés
        # Dimensions : [vocab_size][hidden_size]
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] 
                   for _ in range(len(vocabulary))]
        
        # W2 : Poids entre Neurones Cachés -> Sortie (Prochain Mot)
        # Dimensions : [hidden_size][vocab_size]
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(len(vocabulary))] 
                   for _ in range(hidden_size)]
        
        # Pour stocker les états internes lors de la prédiction (utile pour l'apprentissage)
        self.last_inputs = None
        self.last_hidden_activations = None

    def predict(self, sentence):
        """
        Lit une phrase, active ses neurones et prédit la suite.
        Retourne (meilleur_mot, probabilités, details_debug)
        """
        # 1. Encodage de l'Entrée (Bag of Words simplifié)
        # On regarde quels mots sont présents dans la phrase
        words = sentence.lower().split()
        input_vector = [0.0] * len(self.vocab)
        
        found_words = []
        for w in words:
            idx = self.vocab.get_idx(w)
            if idx is not None:
                input_vector[idx] = 1.0 # Le mot est présent
                found_words.append(w)
        
        self.last_inputs = input_vector

        # 2. Couche Cachée (Les "Neurones" qui "pensent")
        # Chaque neurone caché fait une somme pondérée des entrées + activation
        hidden_activations = []
        for h in range(self.hidden_size):
            # Somme des entrées * poids W1
            weighted_sum = sum(input_vector[i] * self.W1[i][h] for i in range(len(self.vocab)))
            # Activation (Sigmoïde pour avoir une valeur entre 0 et 1)
            activation = sigmoide(weighted_sum)
            hidden_activations.append(activation)
        
        self.last_hidden_activations = hidden_activations

        # 3. Couche de Sortie (Les Mots Candidats)
        # Calcul des scores (logits) pour chaque mot du vocabulaire
        logits = []
        for i in range(len(self.vocab)):
            # Somme des (activations cachées * poids W2)
            z = sum(hidden_activations[h] * self.W2[h][i] for h in range(self.hidden_size))
            logits.append(z)
        
        # 4. Probabilités (Softmax)
        probs = softmax(logits)

        # Préparation des résultats pour l'affichage
        results = []
        for i, p in enumerate(probs):
            results.append((self.vocab.get_word(i), p))
        
        # Tri par probabilité décroissante
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results, hidden_activations

    def learn(self, target_word):
        """
        Apprend : Ajuste les poids pour favoriser 'target_word' vu le contexte précédent.
        """
        target_idx = self.vocab.get_idx(target_word)
        if target_idx is None:
            print(f"Erreur : Le mot cible '{target_word}' n'est pas dans le vocabulaire initial. Ajout impossible en cours de route (simplification).")
            return

        # On veut que la proba du target soit 1, et les autres 0.
        # Simplification extrême de la rétropropagation (Backpropagation)
        
        # 1. Calcul de l'erreur en sortie
        # On refait une passe forward rapide pour avoir les probs actuelles
        # (On utilise les valeurs stockées dans last_hidden_activations)
        
        current_logits = []
        for i in range(len(self.vocab)):
            z = sum(self.last_hidden_activations[h] * self.W2[h][i] for h in range(self.hidden_size))
            current_logits.append(z)
        current_probs = softmax(current_logits)

        # Gradient de l'erreur (pour Cross-Entropy + Softmax, c'est simple: prob - target)
        # target = 1 pour le bon mot, 0 pour les autres
        output_errors = []
        for i in range(len(self.vocab)):
            target_val = 1.0 if i == target_idx else 0.0
            error = current_probs[i] - target_val # Positif si on surestime, Négatif si on sous-estime
            output_errors.append(error)

        # 2. Mise à jour des poids W2 (Caché -> Sortie)
        # w_new = w_old - learning_rate * error * input_activation
        for h in range(self.hidden_size):
            for i in range(len(self.vocab)):
                gradient = output_errors[i] * self.last_hidden_activations[h]
                self.W2[h][i] -= self.learning_rate * gradient

        # 3. Calcul de l'erreur propagée à la couche cachée
        hidden_errors = []
        for h in range(self.hidden_size):
            error_sum = sum(output_errors[i] * self.W2[h][i] for i in range(len(self.vocab)))
            # Dérivée de l'activation (sigmoid)
            derivative = derivee_sigmoide(self.last_hidden_activations[h])
            hidden_errors.append(error_sum * derivative)

        # 4. Mise à jour des poids W1 (Entrée -> Caché)
        for i in range(len(self.vocab)):
            if self.last_inputs[i] > 0: # Optimisation: on ne met à jour que si l'entrée était active
                for h in range(self.hidden_size):
                    gradient = hidden_errors[h] * self.last_inputs[i]
                    self.W1[i][h] -= self.learning_rate * gradient

    def add_new_word(self, new_word):
        """Ajoute un nouveau mot au cerveau et redimensionne les matrices de poids."""
        if self.vocab.get_idx(new_word) is not None:
            return # Déjà connu
        
        # 1. Ajouter au vocabulaire
        self.vocab.add(new_word)
        
        # 2. Agrandir W1 (Entrée -> Caché)
        # On ajoute une nouvelle ligne de poids connectant ce nouveau mot (entrée) aux neurones cachés
        # Dimensions W1 : [vocab_size][hidden_size] -> [vocab_size + 1][hidden_size]
        new_w1_row = [random.uniform(-0.5, 0.5) for _ in range(self.hidden_size)]
        self.W1.append(new_w1_row)
        
        # 3. Agrandir W2 (Caché -> Sortie)
        # On ajoute une nouvelle colonne de poids connectant les neurones cachés à ce nouveau mot (sortie)
        # Dimensions W2 : [hidden_size][vocab_size] -> [hidden_size][vocab_size + 1]
        for h in range(self.hidden_size):
            self.W2[h].append(random.uniform(-0.5, 0.5))
            
        # 4. Agrandir last_inputs si nécessaire (pour que learn() fonctionne juste après)
        if self.last_inputs is not None:
             self.last_inputs.append(0.0) # Le nouveau mot n'était pas dans l'entrée précédente

        print(f"   [Neuroplasticité] J'ai créé de nouvelles connexions synaptiques pour le mot '{new_word}'.")

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "W1": self.W1,
            "W2": self.W2
        }

    @staticmethod
    def from_dict(data, vocab):
        llm = PedagogicalLLM(vocab, hidden_size=data["hidden_size"])
        llm.W1 = data["W1"]
        llm.W2 = data["W2"]
        return llm

# --- 4. Persistance (Sauvegarde/Chargement) ---

BRAIN_FILE = "brain_data.json"

def save_brain(llm):
    data = {
        "vocab": llm.vocab.to_list(),
        "brain": llm.to_dict()
    }
    try:
        with open(BRAIN_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"   [Sauvegarde] Cerveau sauvegardé dans '{BRAIN_FILE}'")
    except Exception as e:
        print(f"   [Erreur] Impossible de sauvegarder : {e}")

def load_brain():
    if not os.path.exists(BRAIN_FILE):
        return None
    
    try:
        print(f"   [Chargement] Lecture de '{BRAIN_FILE}'...")
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        vocab = SimpleVocabulary.from_list(data["vocab"])
        llm = PedagogicalLLM.from_dict(data["brain"], vocab)
        return llm
    except Exception as e:
        print(f"   [Erreur] Fichier corrompu ou illisible ({e}). On repart à zéro.")
        return None

# --- 5. Programme Principal ---

def main():
    print("=== MON PREMIER LLM NEURONAL PÉDAGOGIQUE (AVEC MÉMOIRE) ===")
    
    # Tentative de chargement
    llm = load_brain()

    if llm is None:
        print("Initialisation d'un nouveau cerveau...")
        # 1. Création du vocabulaire de base
        vocab = SimpleVocabulary()
        initial_words = ["paris", "londres", "madrid", "est", "une", "ville", "en", "france", "angleterre", "espagne", "beau", "grand", "monde", "le", "la"]
        for w in initial_words:
            vocab.add(w)
        
        # 2. Création du LLM
        nb_neurons = 5
        llm = PedagogicalLLM(vocab, hidden_size=nb_neurons)
        print(f"Cerveau créé avec {nb_neurons} neurones.")
    else:
        print("Cerveau restauré avec succès !")
        print(f"Vocabulaire : {len(llm.vocab)} mots connu.")

    while True:
        print("\n" + "="*50)
        sentence = input("\nEntrez le début d'une phrase (ou 'q' pour quitter)\n> ")
        if sentence.lower() == 'q':
            break
        
        # Prédiction
        candidates, activations = llm.predict(sentence)
        top_word, top_prob = candidates[0]

        # Affichage Pédagogique
        print(f"\n[Analyse Neuronale]")
        print(f"Activations de la couche cachée : {[f'{a:.2f}' for a in activations]}")
        print("Note : Une valeur proche de 1.0 signifie que le neurone est très actif (il a repéré quelque chose).")
        
        print(f"\n[Prédictions]")
        for word, prob in candidates[:3]: # Top 3
            print(f" - '{word}' : {prob*100:.1f}%")
        
        print(f"\n>>> Le modèle complète par : '{top_word}'")

        # Vérification / Apprentissage
        user_feedback = input(f"Est-ce correct ? (Entrée pour OUI, ou écrivez le VRAI mot pour corriger) : ")
        
        if user_feedback.strip() == "":
            print("Super ! Je renforce mes connexions pour la prochaine fois.")
            # On lui fait apprendre que c'était le bon choix (renforcement positif)
            llm.learn(top_word)
            save_brain(llm)
        else:
            correct_word = user_feedback.strip().lower()
            
            # Gestion des mots inconnus
            if llm.vocab.get_idx(correct_word) is None:
                print(f"Tiens, je ne connais pas le mot '{correct_word}'.")
                llm.add_new_word(correct_word)

            print(f"Ah d'accord ! Je devais dire '{correct_word}'.")
            print(">>> J'apprends de mon erreur... (Modification des poids synaptiques en cours)")
            
            # Boucle d'apprentissage rapide pour bien retenir (Overfitting volontaire pour la démo)
            for _ in range(5): 
                llm.learn(correct_word)
            
            save_brain(llm)

            # Vérification immédiate
            new_candidates, _ = llm.predict(sentence)
            new_top, new_prob = new_candidates[0]
            print(f"Correction appliquée. Maintenant pour cette phrase, je prédis '{new_top}' à {new_prob*100:.1f}%.")

if __name__ == "__main__":
    main()
