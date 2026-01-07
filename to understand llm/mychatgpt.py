import math
import random
import json
import os
import time

# --- 1. Outils Mathématiques ---

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    sum_exps = sum(exps)
    probs = [e / sum_exps for e in exps]
    return probs

def sigmoide(x):
    return 1 / (1 + math.exp(-x))

def derivee_sigmoide(output):
    return output * (1 - output)

# --- 2. Vocabulaire ---

def tokenize(text):
    """Sépare la ponctuation des mots pour une meilleure compréhension."""
    # On ajoute des espaces autour de la ponctuation
    for p in [".", ",", "!", "?", "'", ":", ";", "-"]:
        text = text.replace(p, f" {p} ")
    return text.split()

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    
    def add(self, word):
        word = word.lower()
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

    @staticmethod
    def load_from_file(filepath):
        v = Vocabulary()
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                words = json.load(f)
            for w in words:
                v.add(w)
        return v

# --- 3. Configuration "façon Llama" (Version Pédagogique) ---

class ModelConfig:
    """
    Ces paramètres imitent ceux que vous voyez dans les logs d'Ollama (Llama 3),
    mais avec des valeurs réduites pour tourner sur un simple script Python.
    """
    vocab_size = 0        # Sera défini au chargement (Llama: 128256)
    embedding_length = 32 # "taille du neurone", richesse sémantique (Llama: 3072)
    block_count = 2       # Nombre d'étages de neurones "Layers" (Llama: 28)
    context_length = 16   # Nb de mots max en mémoire (Llama: 131072 !)
    head_count = 1        # Simplifié ici (Llama: 24 têtes d'attention)
    learning_rate = 0.1

# --- 4. Le Cerveau (Structure LLM Simplifiée) ---

class MyChatGPT:
    def __init__(self, vocabulary, config=None):
        self.vocab = vocabulary
        
        # Configuration
        if config is None:
            self.config = ModelConfig()
        else:
            self.config = config
        self.config.vocab_size = len(vocabulary)
        
        print(f"   [Init] Architecture: {self.config.block_count} blocs, {self.config.embedding_length} neurones/bloc.")

        # 1. TOKEN EMBEDDINGS (La Carte d'Identité sémantique)
        # Au lieu de '0' et '1', chaque mot a un vecteur unique de nombres.
        # Log: llama.embedding_length
        self.token_embedding_table = [[random.uniform(-0.1, 0.1) for _ in range(self.config.embedding_length)] 
                                      for _ in range(self.config.vocab_size)]

        # 2. BLOCKS (Le Cerveau Profond)
        # Une série de couches de neurones. Plus il y en a, plus le modèle est "profond" (Deep Learning).
        # Log: llama.block_count
        self.blocks = []
        for i in range(self.config.block_count):
            # Chaque bloc transforme un vecteur d'embedding en un autre vecteur
            # Matrice Weights: [Input_Dim][Output_Dim]
            block_W = [[random.uniform(-0.1, 0.1) for _ in range(self.config.embedding_length)] 
                       for _ in range(self.config.embedding_length)]
            self.blocks.append(block_W)

        # 3. LM HEAD (La Sortie)
        # Convertit la pensée abstraite finale en probabilités pour chaque mot du dictionnaire.
        # Matrice: [Embedding_Dim][Vocab_Size]
        self.lm_head = [[random.uniform(-0.1, 0.1) for _ in range(self.config.vocab_size)] 
                        for _ in range(self.config.embedding_length)]
        
        # Mémoire cache pour la rétropropagation (Apprentissage)
        self.cache = {} 

    def predict_next_word(self, context_words):
        # 0. Gestion du contexte (On garde seulement les N derniers mots)
        # Log: llama.context_length
        if len(context_words) > self.config.context_length:
            context_words = context_words[-self.config.context_length:]

        # 1. EMBEDDING & ATTENTION SIMPLIFIÉE
        # Dans Llama, l'Attention est complexe. Ici, on fait une simple moyenne du contexte.
        # On calcule le vecteur "Sens global de la phrase".
        current_vector = [0.0] * self.config.embedding_length
        valid_words_count = 0
        
        for w in context_words:
            idx = self.vocab.get_idx(w)
            if idx is not None:
                # On additionne le sens de chaque mot
                emb = self.token_embedding_table[idx]
                for i in range(self.config.embedding_length):
                    current_vector[i] += emb[i]
                valid_words_count += 1
        
        # Sauvegarde pour l'apprentissage (Input de la couche 0)
        self.cache['layer_inputs'] = [current_vector[:]] # Liste qui stockera l'input de chaque bloc
        self.cache['layer_activations'] = [] # Liste qui stockera la sortie de chaque bloc

        # 2. PASSAGE DANS LES BLOCS (Forward Pass)
        for block_idx, block_W in enumerate(self.blocks):
            next_vector = []
            
            # Calcul Neuronal Classique (Produit Matriciel + Activation)
            for out_dim in range(self.config.embedding_length):
                weighted_sum = sum(current_vector[in_dim] * block_W[in_dim][out_dim] 
                                   for in_dim in range(self.config.embedding_length))
                next_vector.append(sigmoide(weighted_sum))
            
            # On stocke pour l'apprentissage
            self.cache['layer_activations'].append(next_vector)
            # L'input du prochain bloc est la sortie de celui-ci (Deep Learning)
            self.cache['layer_inputs'].append(next_vector[:]) 
            
            current_vector = next_vector

        # 3. SORTIE (LM Head)
        logits = []
        final_thought = current_vector
        for vocab_idx in range(self.config.vocab_size):
            z = sum(final_thought[dim] * self.lm_head[dim][vocab_idx] 
                   for dim in range(self.config.embedding_length))
            logits.append(z)
        
        return softmax(logits)

    def generate_response(self, prompt, max_len=15):
        """Génère une réponse mot par mot."""
        generated_words = []
        context = tokenize(prompt)
        
        print(f"\n[Génération ({len(self.blocks)} layers)...] ", end="", flush=True)
        
        for _ in range(max_len):
            probs = self.predict_next_word(context)
            
            best_idx = probs.index(max(probs))
            next_word = self.vocab.get_word(best_idx)
            
            if generated_words and next_word == generated_words[-1]:
                print(" [Stop: Loop]")
                break
                
            print(f"{next_word}...", end="", flush=True)
            time.sleep(0.05)

            generated_words.append(next_word)
            context.append(next_word)
            
            if next_word in [".", "?", "!"]:
                print(" [Fin]")
                break
        else:
            print(" [Max]")
        
        return " ".join(generated_words)

    def learn(self, target_word):
        """Rétropropagation à travers TOUS les blocs (Backpropagation Through Time/Layers)."""
        target_idx = self.vocab.get_idx(target_word)
        if target_idx is None: return

        # --- 1. Calcul de l'erreur finale ---
        # On refait le calcul de sortie pour être sûr d'avoir les gradés frais
        final_layer_output = self.cache['layer_activations'][-1] # Sortie du dernier bloc
        
        current_logits = []
        for i in range(self.config.vocab_size):
            z = sum(final_layer_output[h] * self.lm_head[h][i] for h in range(self.config.embedding_length))
            current_logits.append(z)
        current_probs = softmax(current_logits)

        # Erreur (Target - Prediction)
        output_errors = []
        for i in range(self.config.vocab_size):
            target_val = 1.0 if i == target_idx else 0.0
            output_errors.append(current_probs[i] - target_val)

        # --- 2. Mise à jour LM Head (Dernière couche) ---
        # On calcule aussi l'erreur à propager vers le bas (vers les blocs)
        error_propagated_to_last_block = [0.0] * self.config.embedding_length
        
        for h in range(self.config.embedding_length):
            for i in range(self.config.vocab_size):
                # Gradient pour le poids
                grad = output_errors[i] * final_layer_output[h]
                self.lm_head[h][i] -= self.config.learning_rate * grad
                
                # Somme des erreurs pour la couche du dessous
                error_propagated_to_last_block[h] += output_errors[i] * (self.lm_head[h][i] + self.config.learning_rate * grad) # Astuce: on utilise le poids avant maj techniquement mais ici approx

        # --- 3. Rétropropagation à travers les BLOCKS (Boucle inversée) ---
        current_layer_error = error_propagated_to_last_block
        
        # On remonte de le dernière couche (block_count - 1) vers la première (0)
        for i in reversed(range(self.config.block_count)):
            input_val = self.cache['layer_inputs'][i] # Ce qui est entré dans ce bloc
            output_val = self.cache['layer_activations'][i] # Ce qui en est sorti
            block_W = self.blocks[i]
            
            prev_layer_error = [0.0] * self.config.embedding_length # Erreur à envoyer au bloc d'en dessous (ou embeddings)
            
            for out_dim in range(self.config.embedding_length):
                # Dérivée de la sigmoide : f'(x) = f(x)(1-f(x))
                derivative = output_val[out_dim] * (1 - output_val[out_dim])
                term = current_layer_error[out_dim] * derivative
                
                for in_dim in range(self.config.embedding_length):
                    # Mise à jour des poids du bloc
                    block_W[in_dim][out_dim] -= self.config.learning_rate * term * input_val[in_dim]
                    
                    # Accumulation erreur pour étage inférieur
                    prev_layer_error[in_dim] += term * block_W[in_dim][out_dim]
            
            current_layer_error = prev_layer_error

        # --- 4. Mise à jour des Embeddings (L'intuition des mots) ---
        # (Optionnel mais puissant : le modèle affine le sens des mots)
        # current_layer_error contient maintenant l'erreur au niveau des embeddings
        # Ce serait complexe ici car il faut savoir quels mots étaient dans le contexte.
        # Pour simplifier pédagogiquement, on laisse les embeddings fixes ou on ferait une simple mise à jour.
        pass 

    
    def from_dict(self, data):
        self.config.block_count = data.get("config", {}).get("block_count", 2)
        self.config.embedding_length = data.get("config", {}).get("embedding_length", 20)
        
        # Reconstitution de la structure
        self.token_embedding_table = data["token_embedding_table"]
        self.blocks = data["blocks"]
        self.lm_head = data["lm_head"]

    def to_dict(self):
        return {
            "config": {
                "block_count": self.config.block_count,
                "embedding_length": self.config.embedding_length,
                "vocab_size": self.config.vocab_size
            },
            "token_embedding_table": self.token_embedding_table,
            "blocks": self.blocks,
            "lm_head": self.lm_head
        }

    def train_on_corpus(self, corpus_text, epochs=20):
        print("   -> Lecture et apprentissage du corpus...")
        sentences = [s.strip() for s in corpus_text.split('.') if s.strip()]
        
        for epoch in range(epochs):
            for sentence in sentences:
                words = tokenize(sentence)
                # Fenêtre glissante : Contexte -> Target
                # Ex: "hello bonjour" -> apprend que "hello" prédit "bonjour"
                for i in range(len(words)-1):
                    ctx = words[:i+1] # Tout le début de phrase sert de contexte
                    target = words[i+1]
                    
                    self.predict_next_word(ctx) # Forward
                    self.learn(target)          # Backward
            
            if epoch % 5 == 0:
                print(f"      Epoch {epoch}/{epochs} terminée.")


# --- 4. Persistance ---

CHAT_BRAIN_FILE = "chat_brain_v2.json"

def save_chat_brain(chatbot):
    data = {
        "brain": chatbot.to_dict()
    }
    try:
        # On sauvegarde au même endroit que le script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, CHAT_BRAIN_FILE)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"   [Sauvegarde] Cerveau sauvegardé dans '{CHAT_BRAIN_FILE}'")
    except Exception as e:
        print(f"   [Erreur] Impossible de sauvegarder : {e}")

def load_chat_brain(chatbot):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, CHAT_BRAIN_FILE)
    
    if not os.path.exists(path):
        return False
    
    try:
        print(f"   [Chargement] Lecture de '{CHAT_BRAIN_FILE}'...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        chatbot.from_dict(data["brain"])
        return True
    except Exception as e:
        print(f"   [Erreur] Fichier corrompu ({e}).")
        return False


# --- 5. Corpus d'Entraînement (La "Connaissance" Initiale) ---

CORPUS_DIALOGUE = """
hello bonjour comment allez vous .
bonjour je vais bien merci .
comment allez vous .
je vais bien merci et vous .
bien merci .
que puis je faire pour vous aujourd'hui .
je suis ravi de l apprendre .
qui est le président .
le président est emmanuel macron .
quelle est la capitale de la france .
paris est la capitale de la france .
au revoir à bientôt .
"""

# --- 5. Programme Principal ---

def main():
    print("=== MyChatGPT (Simulateur LLM) ===")
    
    # 1. Chargement Vocabulaire
    # On récupère le chemin du script actuel pour trouver le json à côté
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(script_dir, "french_vocabulary.json")
    
    vocab = Vocabulary.load_from_file(vocab_path)
    
    if len(vocab) == 0:
        print(f"[ERREUR] Impossible de charger le vocabulaire depuis '{vocab_path}'.")
        return

    print(f"Vocabulaire chargé : {len(vocab)} mots.")
    
    # 2. Création & Entraînement
    chatbot = MyChatGPT(vocab)
    
    # Tentative de chargement
    loaded = load_chat_brain(chatbot)
    
    if not loaded:
        print("Initialisation du cerveau et pré-entraînement sur le corpus de dialogue...")
        chatbot.train_on_corpus(CORPUS_DIALOGUE, epochs=50)
        save_chat_brain(chatbot)
    else:
        print("Cerveau restauré ! Je me souviens de nos conversations.")
    
    print("\n" + "="*50)
    print("IA Prête ! Discutez avec elle (tapez 'q' pour quitter).")
    print("Essayez: 'hello', 'comment allez vous', 'qui est le président'...")
    print("="*50)

    while True:
        user_input = input("\nVous: ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        
        if not user_input:
            continue

        # Le prompt pour l'IA est l'input utilisateur
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")
        
        # Feedback Utilisateur
        print("\n[Feedback] Appuyez sur Entrée si la réponse est bonne.")
        correction = input("Sinon, écrivez ce que j'aurais dû répondre : ").strip()
        
        if correction:
            print(">>> Ah ! J'apprends cette nouvelle tournure...")
            # On construit une phrase d'entraînement: "Question Réponse ."
            # On nettoie un peu pour avoir des espaces
            training_sentence = f"{user_input} {correction} ."
            
            # On surentraîne un peu sur cette correction pour qu'elle s'imprime bien
            # On augmente le nombre d'époques pour bien ancrer l'information
            chatbot.train_on_corpus(training_sentence, epochs=100)
            save_chat_brain(chatbot)
            print(">>> C'est noté pour la prochaine fois !")

if __name__ == "__main__":
    main()
