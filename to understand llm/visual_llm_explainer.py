import tkinter as tk
from tkinter import ttk, font
import random
import time

class PedagogicalLLM:
    """
    Simule un LLM simplifié pour des besoins pédagogiques.
    Utilise à la fois des règles hardcodées pour l'exemple précis
    et un modèle N-gram simple pour le reste.
    """
    def __init__(self):
        self.context_window = 5
        # Données d'entraînement pédagogiques (connaissance "binaire" du modèle)
        self.knowledge = {
            "paris est une ville en": {
                "france": 0.85,
                "europe": 0.10,
                "fête": 0.04,
                "ruine": 0.01
            },
            "le chat mange de la": {
                "viande": 0.60,
                "pâtée": 0.30,
                "salade": 0.05,
                "pierre": 0.05
            },
            "je suis en train de": {
                "manger": 0.40,
                "dormir": 0.30,
                "coder": 0.20,
                "voler": 0.10
            }
        }
        
    def predict(self, text):
        """
        Retourne une liste de tuples (mot, probabilité) pour le prochain mot.
        """
        text = text.lower().strip()
        
        # 1. Vérifie si on a une correspondance exacte dans notre "base de connaissances"
        # On regarde si la fin de la phrase correspond à une clé
        for key, probs in self.knowledge.items():
            if text.endswith(key):
                return sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # 2. Sinon, génération de probabilités aléatoires mais cohérentes pour l'exemple
        # "Hallucination" contrôlée pour la démo si le contexte est inconnu
        fallback_words = ["le", "la", "un", "une", "et", "mais", "pour", "avec"]
        probs = {}
        remaining_prob = 1.0
        
        count = 4
        for i in range(count):
            if i == count - 1:
                prob = remaining_prob
            else:
                prob = round(random.uniform(0.01, remaining_prob * 0.7), 2)
                remaining_prob -= prob
            
            word = random.choice(fallback_words)
            probs[word] = prob
            
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)

    def get_token_id(self, word):
        """Génère un faux ID de token pour la visualisation."""
        return sum(ord(c) for c in word) + 1000

class LLMVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualisateur de LLM - Comment ça marche ?")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")
        
        self.model = PedagogicalLLM()
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Styles ---
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 11))
        
        # --- Header ---
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(fill="x")
        
        title = ttk.Label(header_frame, text="🧠 Au cœur du LLM", font=("Helvetica", 18, "bold"), foreground="#333")
        title.pack()
        subtitle = ttk.Label(header_frame, text="Comprendre la prédiction du prochain mot par probabilités", font=("Helvetica", 10, "italic"))
        subtitle.pack()
        
        # --- Input Section ---
        input_frame = ttk.Frame(self.root, padding="20")
        input_frame.pack(fill="x")
        
        lbl_input = ttk.Label(input_frame, text="Votre phrase (le Contexte) :")
        lbl_input.pack(anchor="w")
        
        self.input_var = tk.StringVar(value="Paris est une ville en ")
        self.entry = ttk.Entry(input_frame, textvariable=self.input_var, font=("Consolas", 14), width=60)
        self.entry.pack(fill="x", pady=5)
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill="x", pady=5)
        
        self.btn_predict = tk.Button(btn_frame, text="🔮 Prédire le prochain mot", command=self.run_prediction, 
                                     bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
        self.btn_predict.pack(side="left", padx=5)
        
        self.btn_clear = tk.Button(btn_frame, text="🗑️ Effacer", command=
                                   lambda: self.input_var.set(""), bg="#f44336", fg="white", padx=10)
        self.btn_clear.pack(side="right", padx=5)

        # --- Visualization Area (Canvas) ---
        vis_frame = ttk.Frame(self.root, padding="10")
        vis_frame.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(vis_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # --- Explanation Box ---
        explain_frame = ttk.LabelFrame(self.root, text="Explication", padding="10")
        explain_frame.pack(fill="x", padx=20, pady=20)
        
        self.lbl_explanation = ttk.Label(explain_frame, text="Entrez une phrase et cliquez sur 'Prédire' pour voir les neurones s'activer !", wraplength=900, justify="left")
        self.lbl_explanation.pack(fill="x")



    def animate_tokenization(self, nodes, words):
        """Anime la transformation Mot -> ID (Token)."""
        self.lbl_explanation.config(text="1. TOKENISATION : Le modèle ne lit pas les mots. Il les convertit en nombres (Tokens) unique.")
        self.root.update()
        time.sleep(1.0)
        
        for i, (x_end, y) in enumerate(nodes):
            word = words[i]
            token_id = self.model.get_token_id(word)
            # Efface le texte précédent (mot)
            # On recrée un texte par dessus, ou on change l'item existant si on avait gardé son ID
            # Pour simplifier, on dessine par dessus une boite et le nouveau texte
            self.canvas.create_rectangle(x_end - 50, y, x_end + 50, y + 40, fill="#FFCCBC", outline="#FF5722", width=2)
            self.canvas.create_text(x_end, y + 20, text=str(token_id), font=("Courier", 12, "bold"), fill="#BF360C")
            self.root.update()
            time.sleep(0.2)

    def animate_data_flow(self, start_nodes, end_nodes, color="#FFC107"):
        """Anime des particules voyageant d'une couche à l'autre."""
        particles = []
        steps = 20
        
        # Création des particules
        for start_node in start_nodes:
            sx, sy = start_node[0], start_node[1]
            for ex, ey, er in end_nodes:
                 if random.random() < 0.1: # On n'anime pas tout
                     p = self.canvas.create_oval(sx-5, sy-5, sx+5, sy+5, fill=color, outline="")
                     particles.append((p, sx, sy, ex, ey))
        
        # Mouvement
        for step in range(steps):
            for p, sx, sy, ex, ey in particles:
                dx = (ex - sx) / steps
                dy = (ey - sy) / steps
                self.canvas.move(p, dx, dy)
            self.root.update()
            time.sleep(0.02)
            
        # Nettoyage
        for p, _, _, _, _ in particles:
            self.canvas.delete(p)

    def draw_context(self, text):
        words = text.strip().split()
        if not words: return
        
        visible_words = words[-5:] 
        
        start_x = 50
        start_y = 100
        spacing_y = 60
        
        self.canvas.create_text(start_x + 50, 40, text="CONTEXTE (Entrée)", font=("Arial", 12, "bold"), fill="#555")
        
        self.context_nodes = [] # Stocke (x_centre, y_centre)
        self.context_connectors = [] # Stocke le point de connexion droite
        
        for i, word in enumerate(visible_words):
            y = start_y + i * spacing_y
            # Dessin du noeud (Boite mot)
            self.canvas.create_rectangle(start_x, y, start_x + 100, y + 40, fill="#E3F2FD", outline="#2196F3", width=2)
            self.canvas.create_text(start_x + 50, y + 20, text=word, font=("Arial", 10, "bold"))
            
            self.context_nodes.append((start_x + 50, y)) # Centre
            self.context_connectors.append((start_x + 100, y + 20)) # Droite
            
        return visible_words

    def run_prediction(self):
        text = self.input_var.get()
        if not text:
            return
            
        self.canvas.delete("all")
        predictions = self.model.predict(text)
        
        # 1. Contexte & Tokenisation
        visible_words = self.draw_context(text)
        self.root.update()
        time.sleep(0.5)
        
        self.animate_tokenization(self.context_nodes, visible_words)
        time.sleep(0.5)
        
        # 2. Traitement Neuronal
        self.draw_neural_processing() # Dessine les neurones cachés statiques
        self.lbl_explanation.config(text="2. TRAITEMENT : Les nombres passent dans le réseau de neurones (calculs mathématiques).")
        self.root.update()
        
        # Animation Flux : Entrée -> Caché
        # On doit récupérer les coords des neurones cachés (définis dans draw_neural_processing, mais on va le faire retourner)
        # Hack: on appelle draw_neural_processing avant pour avoir self.hidden_nodes, mais on l'avait déjà appelé en ligne 146 avant modif
        pass # La logique suit ci-dessous dans la ré-implémentation globale de run_prediction

        # Re-linking data flow properly
        self.animate_data_flow(self.context_connectors, self.hidden_nodes, color="#2196F3")
        
        # Flash Hidden Nodes
        for hx, hy, hr in self.hidden_nodes:
            self.canvas.create_oval(hx, hy, hx + 50, hy + 50, fill="#FFF176", outline="#FBC02D", width=2)
        self.root.update()
        time.sleep(0.2)
        
        # Animation Flux : Caché -> Sortie
        # On a besoin des coords de destination (calculées dans draw_predictions, mais pas encore appelées)
        # On va pré-calculer les positions de sortie
        res_x = 750
        output_targets = []
        for i in range(len(predictions)):
            y = 100 + i * 80 + 20
            output_targets.append((res_x, y, 0)) # 0 padding
            
        # Animation vers la sortie
        self.animate_data_flow(self.hidden_nodes, output_targets, color="#FFEB3B")

        # 3. Prédictions
        self.lbl_explanation.config(text="3. DÉ-TOKENISATION : Le modèle sort des scores pour chaque ID, qu'on reconvertit en mots.")
        self.draw_predictions(predictions)
        
        # 4. Final Explanation
        best_word, best_prob = predictions[0]
        explanation = (
            f"Le modèle a analysé le contexte : '{text.strip()}'.\n"
            f"Il a transformé les mots en nombres, fait ses calculs matriciels, et calculé les probabilités.\n"
            f"Le mot '{best_word}' a gagné avec {int(best_prob*100)}% de chances."
        )
        self.lbl_explanation.config(text=explanation)

    def draw_neural_processing(self):
        # Dessin d'une "couche cachée" symbolique au milieu
        hidden_x = 400
        start_y = 80
        spacing_y = 50
        
        self.canvas.create_text(hidden_x + 25, 40, text="TRAITEMENT (Réseau de neurones)", font=("Arial", 12, "bold"), fill="#555")
        
        self.hidden_nodes = []
        for i in range(6): # 6 neurones symboliques
            y = start_y + i * spacing_y
            self.canvas.create_oval(hidden_x, y, hidden_x + 50, y + 50, fill="#FFEB3B", outline="#FBC02D", width=2)
            self.hidden_nodes.append((hidden_x, y + 25, hidden_x + 50)) # (gauche, centre_y, droite)
            
        # Connexions Contexte -> Caché (tous vers tous, léger chaos)
        for cx, cy in self.context_nodes:
            for hx, hy, hr in self.hidden_nodes:
                # Lignes fines grises pour montrer la complexité (connexions statiques)
                self.canvas.create_line(cx, cy, hx, hy + 25, fill="#e0e0e0", width=1) # +25 pour atteindre le centre Y du hidden node
                
    def draw_predictions(self, predictions):
        res_x = 750
        start_y = 100
        spacing_y = 80
        
        self.canvas.create_text(res_x + 50, 40, text="PRÉDICTIONS (Sortie)", font=("Arial", 12, "bold"), fill="#555")
        
        max_prob = predictions[0][1]
        
        for i, (word, prob) in enumerate(predictions):
            y = start_y + i * spacing_y
            
            # Liens Caché -> Sortie (seulement les forts pour la lisibilité)
            # On simule que l'activation vient des neurones
            width_line = 1 + (prob * 5) # Plus probable = ligne plus épaisse
            color_line = "#4CAF50" if i == 0 else "#9E9E9E"
            
            # On relie quelques neurones cachés au mot
            for hx, hy, hr in self.hidden_nodes:
                if random.random() < 0.3: # Connexion aléatoire pour l'effet visuel
                     self.canvas.create_line(hr, hy, res_x, y + 20, fill=color_line, width=width_line if i == 0 else 1)

            # Barre de probabilité
            bar_width = (prob / max_prob) * 150
            color = "#4CAF50" if i == 0 else "#CFD8DC"
            
            # Cadre mot + barre
            self.canvas.create_rectangle(res_x, y, res_x + bar_width, y + 40, fill=color, outline="")
            self.canvas.create_text(res_x + 10, y + 20, text=f"{word} ({int(prob*100)}%)", anchor="w", font=("Arial", 11, "bold"))
            
            # Bouton "Choisir" interactif (zone cliquable)
            btn_id = self.canvas.create_rectangle(res_x + 160, y + 5, res_x + 230, y + 35, fill="white", outline="#ccc")
            txt_id = self.canvas.create_text(res_x + 195, y + 20, text="Choisir", fill="#333")
            
            # Callback pour ajouter le mot
            self.canvas.tag_bind(btn_id, "<Button-1>", lambda event, w=word: self.append_word(w))
            self.canvas.tag_bind(txt_id, "<Button-1>", lambda event, w=word: self.append_word(w))

    def append_word(self, word):
        current = self.input_var.get()
        self.input_var.set(current + word + " ")
        # Auto-trigger next prediction? Non, laissons l'utilisateur cliquer pour comprendre l'étape par étape.
        self.run_prediction()

if __name__ == "__main__":
    root = tk.Tk()
    app = LLMVisualizerApp(root)
    root.mainloop()
