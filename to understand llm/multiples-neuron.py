import math

def softmax_simulation():
    print("--- Simulation de 3 Neurones (Tokens candidats) ---")
    print("Contexte : Imaginez que les entrées sont des caractéristiques du texte 'Paris est une ville en'")
    
    # 1. Inputs (Contexte)
    inputs = []
    print("\n1. Définition des entrées (le contexte) :")
    try:
        inputs.append(float(input("Entrée 1 (ex: importance du mot 'Paris') : ")))
        inputs.append(float(input("Entrée 2 (ex: importance du mot 'est') : ")))
        inputs.append(float(input("Entrée 3 (ex: importance du mot 'ville') : ")))
    except ValueError:
        print("Erreur : Veuillez entrer des nombres valides.")
        return
    print(f"Inputs (x) = {inputs}")

    # 2. Neurones (Candidats)
    candidates = ["France", "Espagne", "Pomme"]
    logits = []
    
    print("\n2. Calcul des Logits pour chaque candidat :")
    
    for candidate in candidates:
        print(f"\n--- Neurone pour le token '{candidate}' ---")
        try:
            weights = []
            weights.append(float(input(f"Poids 1 pour '{candidate}' (lien avec Entrée 1) : ")))
            weights.append(float(input(f"Poids 2 pour '{candidate}' (lien avec Entrée 2) : ")))
            weights.append(float(input(f"Poids 3 pour '{candidate}' (lien avec Entrée 3) : ")))
            
            bias = float(input(f"Biais pour '{candidate}' : "))
        except ValueError:
            print("Erreur : Veuillez entrer des nombres valides.")
            return
        
        # Linear combination
        z = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + bias
        logits.append(z)
        print(f"-> Logit (z) calculé pour '{candidate}' : {z:.4f}")
        print(f"   (Calcul : {inputs[0]}*{weights[0]} + {inputs[1]}*{weights[1]} + {inputs[2]}*{weights[2]} + {bias})")

    # 3. Softmax
    print("\n3. Application de la fonction Softmax (Transformation en Probabilités) :")
    
    # Exponentials
    exps = [math.exp(z) for z in logits]
    sum_exps = sum(exps)
    
    print("\n-- Étape A : Exponentielle (e^z) --")
    for i, candidate in enumerate(candidates):
        print(f"Exp({candidate}) = e^{logits[i]:.4f} = {exps[i]:.4f}")
        
    print(f"\n-- Étape B : Somme des exponentielles = {sum_exps:.4f} --")
    
    # Probabilities
    probs = [e / sum_exps for e in exps]
    
    print("\n-- Étape C : Probabilités (Exp / Somme) --")
    for i, candidate in enumerate(candidates):
        print(f"Probabilité('{candidate}') = {exps[i]:.4f} / {sum_exps:.4f} = {probs[i]:.4f} ({probs[i]*100:.2f}%)")

    # Conclusion
    best_candidate_index = probs.index(max(probs))
    winner = candidates[best_candidate_index]
    print(f"\n>>> RÉSULTAT : Le modèle choisit '{winner}' avec {probs[best_candidate_index]*100:.2f}% de confiance.")

if __name__ == "__main__":
    softmax_simulation()
