import tkinter as tk
from tkinter import ttk, messagebox
import math
from neuron import calculate_linear_combination

class NeuronSimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulateur de Neurones LLM")
        self.root.geometry("1100x800")

        # --- Configuration Initiale ---
        config_frame = ttk.LabelFrame(root, text="Configuration Initiale", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(config_frame, text="Phrase (Contexte) :").grid(row=0, column=0, sticky="w")
        self.sentence_var = tk.StringVar(value="Paris est une ville en")
        self.sentence_entry = ttk.Entry(config_frame, textvariable=self.sentence_var, width=50)
        self.sentence_entry.grid(row=0, column=1, padx=5)

        ttk.Label(config_frame, text="Nb Candidats :").grid(row=0, column=2, sticky="w")
        self.n_candidates_var = tk.StringVar(value="3")
        self.n_candidates_entry = ttk.Entry(config_frame, textvariable=self.n_candidates_var, width=5)
        self.n_candidates_entry.grid(row=0, column=3, padx=5)

        self.start_btn = ttk.Button(config_frame, text="Lancer / Démarrer Simulation", command=self.setup_simulation_ui)
        self.start_btn.grid(row=0, column=4, padx=10)

        # --- Zone de Simulation (Scrollable) ---
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # To allow scrolling if many neurons
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Variables de stockage pour calcul temps réel
        self.input_vars = []  # Liste de DoubleVar pour les valeurs des mots
        self.words = [] # Liste des mots
        self.candidates_data = [] # Liste de dicts {name_var, bias_var, weight_vars_list, logit_label, prob_bar}

    def setup_simulation_ui(self):
        # 1. Nettoyer l'interface précédente
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.input_vars = []
        self.candidates_data = []
        
        # 2. Récupérer inputs
        sentence = self.sentence_var.get()
        self.words = sentence.split()
        if not self.words:
            messagebox.showerror("Erreur", "Veuillez entrer une phrase.")
            return

        try:
            n_cand = int(self.n_candidates_var.get())
        except ValueError:
            n_cand = 3
        
        # 3. Créer UI Inputs (Valeurs des mots)
        input_frame = ttk.LabelFrame(self.scrollable_frame, text="1. Valeurs des Mots du Contexte (Inputs)", padding=10)
        input_frame.pack(fill="x", pady=5, padx=5)
        
        for i, word in enumerate(self.words):
            frame = ttk.Frame(input_frame)
            frame.pack(side="left", padx=10)
            ttk.Label(frame, text=f"'{word}'").pack()
            
            # Slider
            val_var = tk.DoubleVar(value=1.0)
            scale = ttk.Scale(frame, from_=-5.0, to=5.0, variable=val_var, orient="vertical", length=100, command=lambda v: self.recalculate())
            scale.pack()
            ttk.Label(frame, textvariable=val_var).pack() # Affiche la valeur numérique
            
            self.input_vars.append(val_var)

        # 4. Créer UI Candidats (Neurones)
        candidates_container = ttk.LabelFrame(self.scrollable_frame, text="2. Neurones Candidats (Poids & Biais)", padding=10)
        candidates_container.pack(fill="both", expand=True, pady=10, padx=5)

        default_names = ["France", "Espagne", "Sandwich", "Berlin", "Pomme"]

        for i in range(n_cand):
            c_frame = ttk.LabelFrame(candidates_container, text=f"Neurone {i+1}", padding=5)
            c_frame.pack(fill="x", pady=5)

            # Nom du candidat
            top_row = ttk.Frame(c_frame)
            top_row.pack(fill="x")
            ttk.Label(top_row, text="Mot Candidat :").pack(side="left")
            name_val = default_names[i] if i < len(default_names) else f"Mot{i+1}"
            name_var = tk.StringVar(value=name_val)
            ttk.Entry(top_row, textvariable=name_var, width=15).pack(side="left", padx=5)

            # Biais
            ttk.Label(top_row, text="| Biais :").pack(side="left", padx=10)
            bias_var = tk.DoubleVar(value=0.0)
            bs = ttk.Scale(top_row, from_=-5.0, to=5.0, variable=bias_var, orient="horizontal", length=100, command=lambda v: self.recalculate())
            bs.pack(side="left")
            ttk.Label(top_row, textvariable=bias_var).pack(side="left")

            # Poids pour chaque mot input
            weights_frame = ttk.Frame(c_frame)
            weights_frame.pack(fill="x", pady=5)
            ttk.Label(weights_frame, text="Poids (Associations) :").pack(anchor="w")

            w_vars = []
            for j, word in enumerate(self.words):
                wf = ttk.Frame(weights_frame)
                wf.pack(side="left", padx=5)
                ttk.Label(wf, text=f"-> {word}", font=("Arial", 8)).pack()
                
                # Default logic for variety
                def_w = 0.0
                if i==0 and j == len(self.words)-1: def_w = 1.0 # Last word strong for first cand
                
                w_var = tk.DoubleVar(value=def_w)
                ws = ttk.Scale(wf, from_=-2.0, to=2.0, variable=w_var, orient="vertical", length=80, command=lambda v: self.recalculate())
                ws.pack()
                # Label arrondi
                # astuce pour afficher l'arrondi : on pourrait utiliser trace mais bon le label simple suffit
                
                w_vars.append(w_var)
            
            # Output Visualization Section for this Neuron
            res_frame = ttk.Frame(c_frame)
            res_frame.pack(fill="x", pady=5, side="bottom")
            
            logit_label = tk.StringVar(value="Logit (z): 0.0")
            ttk.Label(res_frame, textvariable=logit_label, font=("Arial", 9, "bold")).pack(side="left", padx=10)
            
            # Progress bar for probability
            prob_bar = ttk.Progressbar(res_frame, orient="horizontal", length=200, mode="determinate", maximum=100)
            prob_bar.pack(side="left", padx=10)
            
            prob_label_var = tk.StringVar(value="0.0%")
            ttk.Label(res_frame, textvariable=prob_label_var).pack(side="left")

            self.candidates_data.append({
                "name": name_var,
                "bias": bias_var,
                "weights": w_vars,
                "logit_lbl": logit_label,
                "prob_bar": prob_bar,
                "prob_lbl": prob_label_var
            })

        # Initial calc
        self.recalculate()

    def recalculate(self):
        # 1. Collect Inputs
        inputs = [v.get() for v in self.input_vars]
        
        logits = []
        
        # 2. Calculate Logits for all candidates
        for cand in self.candidates_data:
            weights = [w.get() for w in cand["weights"]]
            bias = cand["bias"].get()
            
            # Reuse logic from neuron.py
            z = calculate_linear_combination(inputs, weights, bias)
            cand["logit_lbl"].set(f"Logit (z): {z:.2f}")
            logits.append(z)
        
        # 3. Softmax
        if not logits: return
        
        try:
            # Handle potential overflows with max subtraction (stable softmax)
            max_z = max(logits)
            exps = [math.exp(z - max_z) for z in logits]
            sum_exps = sum(exps)
            probs = [e / sum_exps for e in exps]
        except Exception:
            probs = [0.0] * len(logits) # Fallback

        # 4. Update UI
        best_prob = -1
        best_idx = -1

        for i, prob in enumerate(probs):
            pct = prob * 100
            self.candidates_data[i]["prob_bar"]["value"] = pct
            self.candidates_data[i]["prob_lbl"].set(f"{pct:.1f}%")
            
            if prob > best_prob:
                best_prob = prob
                best_idx = i
        
        # Highlight winner (optional - maybe bold name?)
        # For now, simplistic approach is fine.


def main():
    root = tk.Tk()
    app = NeuronSimulationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
