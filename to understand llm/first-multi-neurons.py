import math
from neuron import calculate_linear_combination

def softmax(logits):
    """Calcule les probabilités à partir des logits (scores bruts) via Softmax."""
    # 1. Exponentielles
    exps = [math.exp(z) for z in logits]
    # 2. Somme des exponentielles
    sum_exps = sum(exps)
    # 3. Probabilités
    probs = [e / sum_exps for e in exps]
    return probs

def llm_neuron_simulation():
    print("=== SIMULATION PEDAGOGIQUE : PRÉDICTION DU PROCHAIN MOT ===")
    print("Ce programme illustre comment un LLM (tête de prédiction) choisit le mot suivant.\n")

    # 1. Saisie de la phrase (Contexte)
    sentence = input("Entrez une phrase ou un début de phrase (ex: 'Paris est une ville en') : ")
    words = sentence.split()
    print(f"\nLa phrase a été découpée en {len(words)} tokens (mots) : {words}")
    
    # 2. Attribution des valeurs aux tokens (Inputs)
    print("\nImaginez que chaque mot est converti en une valeur numérique (embedding simplifié).")
    inputs = []
    for word in words:
        try:
            val = float(input(f" - Valeur pour '{word}' (ex: 1.0) : "))
            inputs.append(val)
        except ValueError:
            print(f"   -> Valeur invalide, on utilise 0.0 par défaut pour '{word}'")
            inputs.append(0.0)
    print(f"Inputs (x) = {inputs}")

    # 3. Configuration des Neurones (Candidats)
    print("\n--- Configuration des Neurones (Mots Candidats) ---")
    try:
        n_candidates = int(input("Combien de mots candidats voulez-vous tester (ex: 3) ? "))
    except ValueError:
        n_candidates = 3
        print("-> Entrée invalide, on part sur 3 candidats.")

    candidates = []     # Noms des candidats (ex: France, Espagne...)
    logits = []         # Valeurs z calculées pour chaque candidat

    for i in range(n_candidates):
        print(f"\n--- Candidat n°{i+1} ---")
        cand_name = input("Quel est le mot candidat (ex: France) ? ")
        candidates.append(cand_name)

        print(f"Définissez les poids pour le neurone '{cand_name}'.")
        print("Les poids déterminent l'importance de chaque mot du contexte pour choisir ce candidat.")
        
        weights = []
        for j, word in enumerate(words):
            try:
                w = float(input(f" - Poids pour le lien '{word}' -> '{cand_name}' (ex: 0.5) : "))
                weights.append(w)
            except ValueError:
                print(f"   -> Valeur invalide, on utilise 0.0")
                weights.append(0.0)
        
        try:
            bias = float(input(f" - Biais pour '{cand_name}' (tendance naturelle à être choisi, ex: 0.0) : "))
        except ValueError:
            bias = 0.0
            print("   -> Valeur invalide, on utilise 0.0")

        # Utilisation de la fonction importée de neuron.py
        z = calculate_linear_combination(inputs, weights, bias)
        logits.append(z)
        print(f"-> Score brut (logit z) pour '{cand_name}' : {z:.4f}")

    # 4. Softmax et Résultats
    print("\n--- Résultat Final (Probabilités) ---")
    print("Application de la fonction Softmax sur les logits pour obtenir des pourcentages.")
    
    if not logits:
        print("Aucun candidat défini.")
        return

    probs = softmax(logits)

    print("\nPrédictions du modèle :")
    for i in range(len(candidates)):
        print(f"  {candidates[i]} : {probs[i]*100:.2f}% (Logit: {logits[i]:.2f})")

    # Trouvez le gagnant
    best_idx = probs.index(max(probs))
    print(f"\n>>> Le mot choisi est : '{candidates[best_idx]}'")
    print(f"Phrase complétée : {sentence} {candidates[best_idx]}")

if __name__ == "__main__":
    llm_neuron_simulation()
