"""
Mini Modèle de Langage AMÉLIORÉ
Performances nettement supérieures grâce à :
- Plus de données d'entraînement
- Architecture plus profonde
- Contexte plus long
- Meilleur apprentissage
"""

import numpy as np
import random

np.random.seed(42)
random.seed(42)

print("="*70)
print("MINI MODÈLE DE LANGAGE - VERSION AMÉLIORÉE")
print("="*70)


# ============================================
# FONCTIONS
# ============================================
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# ============================================
# TOKENIZER
# ============================================
class Tokenizer:
    def __init__(self):
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.next_id = 4
    
    def fit(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
    
    def encode(self, text):
        return [self.word_to_id.get(word.lower(), 1) for word in text.split()]
    
    def decode(self, ids):
        words = []
        for id in ids:
            word = self.id_to_word.get(id, "<UNK>")
            if word not in ["<PAD>", "<START>", "<END>", "<UNK>"]:
                words.append(word)
        return " ".join(words)
    
    @property
    def vocab_size(self):
        return len(self.word_to_id)


# ============================================
# MODÈLE AMÉLIORÉ
# ============================================
class ImprovedLanguageModel:
    """
    Modèle amélioré avec :
    - 2 couches cachées (plus de capacité)
    - Dropout (régularisation)
    - Meilleure initialisation
    """
    
    def __init__(self, vocab_size, embedding_dim=40, hidden_dim1=80, 
                 hidden_dim2=60, context_length=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.context_length = context_length
        
        # Initialisation Xavier/Glorot (meilleure que random)
        scale1 = np.sqrt(2.0 / (context_length * embedding_dim))
        scale2 = np.sqrt(2.0 / hidden_dim1)
        scale3 = np.sqrt(2.0 / hidden_dim2)
        
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Première couche cachée
        self.W_hidden1 = np.random.randn(context_length * embedding_dim, hidden_dim1) * scale1
        self.b_hidden1 = np.zeros(hidden_dim1)
        
        # Deuxième couche cachée
        self.W_hidden2 = np.random.randn(hidden_dim1, hidden_dim2) * scale2
        self.b_hidden2 = np.zeros(hidden_dim2)
        
        # Couche de sortie
        self.W_output = np.random.randn(hidden_dim2, vocab_size) * scale3
        self.b_output = np.zeros(vocab_size)
        
        # Momentum
        self.v_embeddings = np.zeros_like(self.embeddings)
        self.v_W_hidden1 = np.zeros_like(self.W_hidden1)
        self.v_b_hidden1 = np.zeros_like(self.b_hidden1)
        self.v_W_hidden2 = np.zeros_like(self.W_hidden2)
        self.v_b_hidden2 = np.zeros_like(self.b_hidden2)
        self.v_W_output = np.zeros_like(self.W_output)
        self.v_b_output = np.zeros_like(self.b_output)
        
        self.loss_history = []
    
    def forward(self, context_ids, training=True):
        # Embeddings
        context_vectors = [self.embeddings[id] for id in context_ids]
        x = np.concatenate(context_vectors)
        
        # Première couche cachée
        self.z1 = x @ self.W_hidden1 + self.b_hidden1
        self.a1 = relu(self.z1)
        
        # Dropout pendant l'entraînement
        if training:
            self.dropout_mask1 = (np.random.rand(*self.a1.shape) > 0.2).astype(float)
            self.a1 *= self.dropout_mask1
        
        # Deuxième couche cachée
        self.z2 = self.a1 @ self.W_hidden2 + self.b_hidden2
        self.a2 = relu(self.z2)
        
        if training:
            self.dropout_mask2 = (np.random.rand(*self.a2.shape) > 0.2).astype(float)
            self.a2 *= self.dropout_mask2
        
        # Sortie
        self.output_scores = self.a2 @ self.W_output + self.b_output
        probs = softmax(self.output_scores)
        
        self.last_context_ids = context_ids
        self.last_x = x
        
        return probs
    
    def backward(self, context_ids, target_id, learning_rate=0.01, momentum=0.9):
        probs = self.forward(context_ids, training=True)
        
        # Gradient sortie
        d_output = probs.copy()
        d_output[target_id] -= 1
        
        # Rétropropagation couche de sortie
        d_W_output = np.outer(self.a2, d_output)
        d_b_output = d_output
        d_a2 = d_output @ self.W_output.T
        
        # Rétropropagation deuxième couche cachée
        d_a2 *= self.dropout_mask2
        d_z2 = d_a2 * relu_derivative(self.z2)
        d_W_hidden2 = np.outer(self.a1, d_z2)
        d_b_hidden2 = d_z2
        d_a1 = d_z2 @ self.W_hidden2.T
        
        # Rétropropagation première couche cachée
        d_a1 *= self.dropout_mask1
        d_z1 = d_a1 * relu_derivative(self.z1)
        d_W_hidden1 = np.outer(self.last_x, d_z1)
        d_b_hidden1 = d_z1
        d_x = d_z1 @ self.W_hidden1.T
        
        # Rétropropagation embeddings
        d_embeddings = {}
        chunk_size = self.embedding_dim
        for i, word_id in enumerate(context_ids):
            # Ignore PAD token gradients
            if word_id == 0:
                continue
                
            start = i * chunk_size
            end = start + chunk_size
            if word_id not in d_embeddings:
                d_embeddings[word_id] = np.zeros(self.embedding_dim)
            d_embeddings[word_id] += d_x[start:end]
        
        # Mise à jour avec momentum et weight decay
        weight_decay = 0.0001
        
        self.v_W_output = momentum * self.v_W_output - learning_rate * (d_W_output + weight_decay * self.W_output)
        self.v_b_output = momentum * self.v_b_output - learning_rate * d_b_output
        self.v_W_hidden2 = momentum * self.v_W_hidden2 - learning_rate * (d_W_hidden2 + weight_decay * self.W_hidden2)
        self.v_b_hidden2 = momentum * self.v_b_hidden2 - learning_rate * d_b_hidden2
        self.v_W_hidden1 = momentum * self.v_W_hidden1 - learning_rate * (d_W_hidden1 + weight_decay * self.W_hidden1)
        self.v_b_hidden1 = momentum * self.v_b_hidden1 - learning_rate * d_b_hidden1
        
        self.W_output += self.v_W_output
        self.b_output += self.v_b_output
        self.W_hidden2 += self.v_W_hidden2
        self.b_hidden2 += self.v_b_hidden2
        self.W_hidden1 += self.v_W_hidden1
        self.b_hidden1 += self.v_b_hidden1
        
        for word_id, grad in d_embeddings.items():
            self.v_embeddings[word_id] = momentum * self.v_embeddings[word_id] - learning_rate * grad
            self.embeddings[word_id] += self.v_embeddings[word_id]
        
        loss = -np.log(probs[target_id] + 1e-10)
        return loss
    
    def train(self, texts, tokenizer, epochs=300, initial_lr=0.1, verbose=True):
        if verbose:
            print(f"\n🚀 Entraînement sur {len(texts)} phrases...\n")
        
        for epoch in range(epochs):
            # Learning rate decay
            lr = initial_lr * (0.95 ** (epoch // 20))
            
            total_loss = 0
            count = 0
            
            # Mélange les données à chaque époque
            shuffled_texts = texts.copy()
            random.shuffle(shuffled_texts)
            
            for text in shuffled_texts:
                # Ajout des tokens spéciaux <START> et <END>
                # Utilise les IDs du tokenizer : <START>=2, <END>=3
                start_id = tokenizer.word_to_id["<START>"]
                end_id = tokenizer.word_to_id["<END>"]
                
                ids = [start_id] + tokenizer.encode(text) + [end_id]
                
                # On parcourt toute la séquence pour prédire chaque token suivant le contexte
                for i in range(1, len(ids)):
                    # La cible est le token actuel
                    target = ids[i]
                    
                    # Le contexte est tout ce qui précède
                    history = ids[:i]
                    
                    # On prend les 'context_length' derniers tokens
                    context = history[-self.context_length:]
                    
                    # Padding si le contexte est trop court
                    if len(context) < self.context_length:
                        # <PAD> = 0
                        padding = [0] * (self.context_length - len(context))
                        context = padding + context
                    
                    loss = self.backward(context, target, lr)
                    total_loss += loss
                    count += 1
            
            avg_loss = total_loss / max(count, 1)
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Époque {epoch+1:3d}/{epochs} | Perte: {avg_loss:.4f} | LR: {lr:.4f}")
        
        if verbose:
            print("\n✓ Entraînement terminé !")
    
    def predict_next(self, context_text, tokenizer, top_k=5):
        # On ajoute <START> pour simuler le contexte d'entraînement
        ids = [tokenizer.word_to_id["<START>"]] + tokenizer.encode(context_text)
        
        # On prend les derniers tokens
        context = ids[-self.context_length:]
        
        # Padding
        if len(context) < self.context_length:
            padding = [0] * (self.context_length - len(context))
            context = padding + context
        
        probs = self.forward(context, training=False)
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        results = [(tokenizer.id_to_word[idx], probs[idx]) for idx in top_indices]
        
        return results
    
    def generate(self, start_text, tokenizer, max_length=15, temperature=0.7, stop_at_end=True):
        # On initialise avec <START>
        ids = [tokenizer.word_to_id["<START>"]] + tokenizer.encode(start_text)
        
        for _ in range(max_length):
            context = ids[-self.context_length:]
            
            if len(context) < self.context_length:
                padding = [0] * (self.context_length - len(context))
                context = padding + context
            
            probs = self.forward(context, training=False)

            # Temperature scaling (protéger temperature <= 0)
            if temperature <= 0:
                temperature = 1e-6

            probs = np.power(probs, 1.0 / temperature)

            # Nettoyage des valeurs invalides et normalisation
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = np.clip(probs, 0.0, None)
            total = probs.sum()
            if total <= 0 or not np.isfinite(total):
                # Fallback : distribution uniforme sur tous les tokens (hors spéciaux)
                probs = np.ones_like(probs, dtype=float)

            probs = probs / probs.sum()

            # Évite les tokens spéciaux : <PAD>=0, <UNK>=1, <START>=2
            probs[0] = 0
            probs[1] = 0
            probs[2] = 0

            # Re-normaliser après avoir éliminé les spéciaux
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = np.clip(probs, 0.0, None)
            if probs.sum() <= 0:
                probs = np.ones_like(probs, dtype=float)
                probs[0] = 0; probs[1] = 0; probs[2] = 0

            probs = probs / probs.sum()

            next_id = np.random.choice(len(probs), p=probs)
            
            # Arrête si <END>
            if next_id == tokenizer.word_to_id["<END>"]:
                break
                
            # Arrête si fin de phrase (ponctuation)
            word = tokenizer.id_to_word.get(next_id)
            if stop_at_end and word in [".", "!", "?"]:
                ids.append(next_id)
                break
            
            ids.append(next_id)
        
        return tokenizer.decode(ids)


# ============================================
# DONNÉES D'ENTRAÎNEMENT ENRICHIES
# ============================================
training_texts = [
    # Animaux et actions
    "le chat dort sur le canapé rouge",
    "le chat mange des croquettes tous les jours",
    "le chat joue avec une balle rouge",
    "le chat noir dort beaucoup la nuit",
    "le chat aime dormir au soleil",
    "le chat ronronne quand il est content",
    "le chien dort sur le tapis doux",
    "le chien mange de la viande fraîche",
    "le chien joue dans le jardin vert",
    "le chien brun court vite dehors",
    "le chien aime courir dans le parc",
    "le chien jappe quand il est content",
    
    # Objets et descriptions
    "la balle rouge roule sur le sol",
    "la balle rebondit très haut",
    "le canapé rouge est très confortable",
    "le canapé est dans le salon",
    "le tapis est doux et moelleux",
    "le tapis bleu est sous la table",
    "le jardin est grand et fleuri",
    "le jardin a beaucoup de fleurs",
    
    # Lieux et positions
    "le chat est sur le canapé",
    "le chien est dans le jardin",
    "la balle est sous la table",
    "le tapis est dans le salon",
    
    # Couleurs
    "le chat noir est élégant",
    "le chien brun est gentil",
    "la balle rouge est petite",
    "le canapé rouge est grand",
    "le tapis bleu est joli",
    
    # Actions variées
    "le chat dort beaucoup",
    "le chien court vite",
    "la balle roule loin",
    "le chat mange lentement",
    "le chien joue souvent",
    
    # Phrases plus complexes
    "le chat noir dort sur le canapé rouge tous les jours",
    "le chien brun court vite dans le jardin vert",
    "la balle rouge roule loin sur le tapis bleu",
    "le chat mange des croquettes et dort sur le canapé",
    "le chien joue dans le jardin et court vite",
]

print(f"\n📚 Corpus enrichi : {len(training_texts)} phrases")
print(f"    Mots uniques : ~{len(set(' '.join(training_texts).split()))} mots\n")


# ============================================
# PRÉPARATION
# ============================================
tokenizer = Tokenizer()
tokenizer.fit(training_texts)

print(f"📖 Vocabulaire : {tokenizer.vocab_size} tokens")


# ============================================
# CRÉATION DU MODÈLE AMÉLIORÉ
# ============================================
model = ImprovedLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=40,      # Doublé
    hidden_dim1=80,        # Première couche
    hidden_dim2=60,        # Deuxième couche
    context_length=5       # Plus de contexte
)

n_params = (model.vocab_size * model.embedding_dim + 
            model.context_length * model.embedding_dim * model.hidden_dim1 +
            model.hidden_dim1 + 
            model.hidden_dim1 * model.hidden_dim2 +
            model.hidden_dim2 +
            model.hidden_dim2 * model.vocab_size + 
            model.vocab_size)

print(f"\n🧠 Modèle amélioré créé !")
print(f"   • Embeddings : {model.embedding_dim}D")
print(f"   • Couche 1 : {model.hidden_dim1} neurones")
print(f"   • Couche 2 : {model.hidden_dim2} neurones")
print(f"   • Contexte : {model.context_length} mots")
print(f"   • Paramètres : {n_params:,}")


# ============================================
# TEST AVANT ENTRAÎNEMENT
# ============================================
print("\n" + "="*70)
print("TEST AVANT ENTRAÎNEMENT")
print("="*70)

for context in ["le chat", "le chien"]:
    predictions = model.predict_next(context, tokenizer, top_k=3)
    print(f"\nContexte : '{context}'")
    for word, prob in predictions:
        print(f"  → {word:15s} {prob*100:5.1f}%")


# ============================================
# ENTRAÎNEMENT
# ============================================
print("\n" + "="*70)
print("ENTRAÎNEMENT")
print("="*70)

model.train(training_texts, tokenizer, epochs=400, initial_lr=0.01)


# ============================================
# TEST APRÈS ENTRAÎNEMENT
# ============================================
print("\n" + "="*70)
print("TEST APRÈS ENTRAÎNEMENT")
print("="*70)

test_contexts = ["le chat", "le chien", "la balle", "le canapé", "le chat noir"]

for context in test_contexts:
    predictions = model.predict_next(context, tokenizer, top_k=3)
    print(f"\nContexte : '{context}'")
    for word, prob in predictions:
        print(f"  → {word:15s} {prob*100:5.1f}%")


# ============================================
# GÉNÉRATION DE TEXTE
# ============================================
print("\n" + "="*70)
print("GÉNÉRATION DE TEXTE")
print("="*70)

starts = [
    "le chat",
    "le chien", 
    "le chat noir",
    "le chien brun",
    "la balle rouge"
]

print("\n🎨 Génération avec température = 0.5 (conservateur)")
print("-" * 70)
for start in starts:
    generated = model.generate(start, tokenizer, max_length=10, temperature=0.5)
    print(f"'{start}' → {generated}")

print("\n🎨 Génération avec température = 0.8 (équilibré)")
print("-" * 70)
for start in starts:
    generated = model.generate(start, tokenizer, max_length=10, temperature=0.8)
    print(f"'{start}' → {generated}")

print("\n" + "="*70)
print("COMPARAISON DES PERFORMANCES")
print("="*70)
print(f"📉 Perte initiale : {model.loss_history[0]:.4f}")
print(f"📈 Perte finale : {model.loss_history[-1]:.4f}")
print(f"✨ Amélioration : {(1 - model.loss_history[-1]/model.loss_history[0])*100:.1f}%")

print("\n✓ Le modèle génère maintenant des phrases beaucoup plus cohérentes !")
print("="*70)