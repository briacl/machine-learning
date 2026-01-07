"""
Mini Modèle de Langage avec Apprentissage
Script Python exécutable directement
"""

import numpy as np
import random
import matplotlib.pyplot as plt

# Configuration pour des résultats reproductibles
np.random.seed(42)
random.seed(42)

print("="*60)
print("MINI MODÈLE DE LANGAGE AVEC APPRENTISSAGE")
print("="*60)


# ============================================
# FONCTIONS D'ACTIVATION
# ============================================
def softmax(x):
    """Convertit des scores en probabilités"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def relu(x):
    """Fonction d'activation ReLU"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Dérivée de ReLU pour la rétropropagation"""
    return (x > 0).astype(float)


# ============================================
# TOKENIZER
# ============================================
class Tokenizer:
    """Convertit le texte en nombres et vice-versa"""
    
    def __init__(self):
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.next_id = 2
    
    def fit(self, texts):
        """Construit le vocabulaire à partir des textes"""
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
    
    def encode(self, text):
        """Texte → Liste de nombres"""
        return [self.word_to_id.get(word.lower(), 1) for word in text.split()]
    
    def decode(self, ids):
        """Liste de nombres → Texte"""
        return " ".join([self.id_to_word.get(id, "<UNK>") for id in ids])
    
    @property
    def vocab_size(self):
        return len(self.word_to_id)


# ============================================
# MODÈLE DE LANGAGE
# ============================================
class MiniLanguageModel:
    """
    Modèle de langage neuronal qui apprend à prédire le prochain mot
    Architecture : Embedding → Hidden Layer → Output
    """
    
    def __init__(self, vocab_size, embedding_dim=20, hidden_dim=30, context_length=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        
        # Initialisation des poids
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.W_hidden = np.random.randn(context_length * embedding_dim, hidden_dim) * 0.1
        self.b_hidden = np.zeros(hidden_dim)
        self.W_output = np.random.randn(hidden_dim, vocab_size) * 0.1
        self.b_output = np.zeros(vocab_size)
        
        # Momentum
        self.v_embeddings = np.zeros_like(self.embeddings)
        self.v_W_hidden = np.zeros_like(self.W_hidden)
        self.v_b_hidden = np.zeros_like(self.b_hidden)
        self.v_W_output = np.zeros_like(self.W_output)
        self.v_b_output = np.zeros_like(self.b_output)
        
        self.loss_history = []
    
    def forward(self, context_ids):
        """Propagation avant : calcule les probabilités du prochain mot"""
        # 1. Récupère et concatène les embeddings
        context_vectors = [self.embeddings[id] for id in context_ids]
        x = np.concatenate(context_vectors)
        
        # 2. Couche cachée avec ReLU
        self.hidden_input = x @ self.W_hidden + self.b_hidden
        self.hidden_output = relu(self.hidden_input)
        
        # 3. Couche de sortie
        self.output_scores = self.hidden_output @ self.W_output + self.b_output
        
        # 4. Softmax → probabilités
        probs = softmax(self.output_scores)
        
        # Sauvegarde pour la rétropropagation
        self.last_context_ids = context_ids
        self.last_x = x
        
        return probs
    
    def backward(self, context_ids, target_id, learning_rate=0.01, momentum=0.9):
        """Rétropropagation : ajuste les poids pour réduire l'erreur"""
        probs = self.forward(context_ids)
        
        # Gradient de la perte
        d_output = probs.copy()
        d_output[target_id] -= 1
        
        # Rétropropagation couche de sortie
        d_W_output = np.outer(self.hidden_output, d_output)
        d_b_output = d_output
        d_hidden_output = d_output @ self.W_output.T
        
        # Rétropropagation couche cachée
        d_hidden_input = d_hidden_output * relu_derivative(self.hidden_input)
        d_W_hidden = np.outer(self.last_x, d_hidden_input)
        d_b_hidden = d_hidden_input
        d_x = d_hidden_input @ self.W_hidden.T
        
        # Rétropropagation embeddings
        d_embeddings = {}
        chunk_size = self.embedding_dim
        for i, word_id in enumerate(context_ids):
            start = i * chunk_size
            end = start + chunk_size
            if word_id not in d_embeddings:
                d_embeddings[word_id] = np.zeros(self.embedding_dim)
            d_embeddings[word_id] += d_x[start:end]
        
        # Mise à jour avec momentum
        self.v_W_output = momentum * self.v_W_output - learning_rate * d_W_output
        self.v_b_output = momentum * self.v_b_output - learning_rate * d_b_output
        self.v_W_hidden = momentum * self.v_W_hidden - learning_rate * d_W_hidden
        self.v_b_hidden = momentum * self.v_b_hidden - learning_rate * d_b_hidden
        
        self.W_output += self.v_W_output
        self.b_output += self.v_b_output
        self.W_hidden += self.v_W_hidden
        self.b_hidden += self.v_b_hidden
        
        for word_id, grad in d_embeddings.items():
            self.v_embeddings[word_id] = momentum * self.v_embeddings[word_id] - learning_rate * grad
            self.embeddings[word_id] += self.v_embeddings[word_id]
        
        loss = -np.log(probs[target_id] + 1e-10)
        return loss
    
    def train(self, texts, tokenizer, epochs=100, learning_rate=0.01, verbose=True):
        """Entraîne le modèle"""
        if verbose:
            print(f"\n🚀 Entraînement sur {len(texts)} phrases...\n")
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            for text in texts:
                ids = tokenizer.encode(text)
                
                for i in range(len(ids) - self.context_length):
                    context = ids[i:i+self.context_length]
                    target = ids[i+self.context_length]
                    
                    loss = self.backward(context, target, learning_rate)
                    total_loss += loss
                    count += 1
            
            avg_loss = total_loss / max(count, 1)
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Époque {epoch+1:3d}/{epochs} | Perte: {avg_loss:.4f}")
        
        if verbose:
            print("\n✓ Entraînement terminé !")
    
    def predict_next(self, context_text, tokenizer, top_k=5):
        """Prédit les K mots les plus probables"""
        ids = tokenizer.encode(context_text)[-self.context_length:]
        
        while len(ids) < self.context_length:
            ids.insert(0, 0)
        
        probs = self.forward(ids)
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        results = [(tokenizer.id_to_word[idx], probs[idx]) for idx in top_indices]
        
        return results
    
    def generate(self, start_text, tokenizer, length=10, temperature=0.8):
        """Génère du texte"""
        ids = tokenizer.encode(start_text)
        
        for _ in range(length):
            context = ids[-self.context_length:]
            while len(context) < self.context_length:
                context.insert(0, 0)
            
            probs = self.forward(context)
            
            # Temperature scaling
            probs = np.power(probs, 1/temperature)
            probs = probs / probs.sum()
            
            next_id = np.random.choice(len(probs), p=probs)
            ids.append(next_id)
        
        return tokenizer.decode(ids)


# ============================================
# DONNÉES D'ENTRAÎNEMENT
# ============================================
training_texts = [
    "le chat dort sur le canapé rouge",
    "le chat mange des croquettes",
    "le chat joue avec une balle",
    "le chien dort sur le tapis",
    "le chien mange de la viande",
    "le chien joue dans le jardin",
    "le chat noir dort beaucoup",
    "le chien brun court vite",
    "la balle rouge roule",
    "le canapé rouge est confortable",
    "le jardin est grand",
    "le tapis est doux"
]

print(f"\n📚 Corpus : {len(training_texts)} phrases")


# ============================================
# PRÉPARATION
# ============================================
print("\n" + "="*60)
print("1. TOKENISATION")
print("="*60)

tokenizer = Tokenizer()
tokenizer.fit(training_texts)

print(f"\n📖 Vocabulaire : {tokenizer.vocab_size} mots")
print(f"Mots : {list(tokenizer.word_to_id.keys())[:10]}...")


# ============================================
# CRÉATION DU MODÈLE
# ============================================
print("\n" + "="*60)
print("2. CRÉATION DU MODÈLE")
print("="*60)

model = MiniLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=15,
    hidden_dim=25,
    context_length=3
)

print(f"\n🧠 Modèle créé !")
print(f"  • Embeddings : {model.embedding_dim}D")
print(f"  • Couche cachée : {model.hidden_dim} neurones")
print(f"  • Contexte : {model.context_length} mots")


# ============================================
# TEST AVANT ENTRAÎNEMENT
# ============================================
print("\n" + "="*60)
print("3. TEST AVANT ENTRAÎNEMENT")
print("="*60)

test_context = "le chat"
predictions = model.predict_next(test_context, tokenizer, top_k=3)
print(f"\nContexte : '{test_context}'")
print("Prédictions (modèle non entraîné) :")
for word, prob in predictions:
    print(f"  {word:15s} {prob*100:5.1f}%")


# ============================================
# ENTRAÎNEMENT
# ============================================
print("\n" + "="*60)
print("4. ENTRAÎNEMENT")
print("="*60)

model.train(training_texts, tokenizer, epochs=200, learning_rate=0.05)


# ============================================
# TEST APRÈS ENTRAÎNEMENT
# ============================================
print("\n" + "="*60)
print("5. TEST APRÈS ENTRAÎNEMENT")
print("="*60)

test_cases = ["le chat", "le chien", "sur le"]

for test in test_cases:
    predictions = model.predict_next(test, tokenizer, top_k=3)
    print(f"\nContexte : '{test}'")
    print("Prédictions :")
    for word, prob in predictions:
        print(f"  {word:15s} {prob*100:5.1f}%")


# ============================================
# GÉNÉRATION DE TEXTE
# ============================================
print("\n" + "="*60)
print("6. GÉNÉRATION DE TEXTE")
print("="*60)

for start in ["le chat", "le chien", "la balle"]:
    generated = model.generate(start, tokenizer, length=7, temperature=0.6)
    print(f"\nDébut : '{start}'")
    print(f"Généré : '{generated}'")


# ============================================
# VISUALISATION
# ============================================
print("\n" + "="*60)
print("7. VISUALISATION DE L'APPRENTISSAGE")
print("="*60)

plt.figure(figsize=(10, 5))
plt.plot(model.loss_history, linewidth=2, color='#2563eb')
plt.title('Évolution de la perte pendant l\'entraînement', fontsize=14, fontweight='bold')
plt.xlabel('Époque', fontsize=12)
plt.ylabel('Perte (Cross-Entropy)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=150)
print("\n📊 Graphique sauvegardé : training_loss.png")

print(f"\n📉 Perte initiale : {model.loss_history[0]:.4f}")
print(f"📈 Perte finale : {model.loss_history[-1]:.4f}")
print(f"✨ Amélioration : {(1 - model.loss_history[-1]/model.loss_history[0])*100:.1f}%")


# ============================================
# INTERACTION
# ============================================
print("\n" + "="*60)
print("8. MODE INTERACTIF")
print("="*60)
print("\nVous pouvez maintenant tester le modèle !")
print("Exemples de commandes à essayer :\n")
print("  model.generate('le chat', tokenizer, length=8)")
print("  model.predict_next('le chien', tokenizer)")
print("\nOu modifiez le code pour expérimenter avec différents paramètres !")

print("\n" + "="*60)
print("✓ TERMINÉ !")
print("="*60)