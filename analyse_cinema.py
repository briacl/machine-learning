import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def pedagogie_ml(etape, message):
    """Affiche un message pédagogique formaté."""
    print(f"\n[ML Step {etape}] {message}")
    print("-" * 50)

def main():
    # --- ÉTAPE 1 : CHARGEMENT DES DONNÉES ---
    pedagogie_ml(1, "Chargement et préparation des données (X et y)")
    print("Nous chargeons le fichier 'cinema.csv'...")
    try:
        df = pd.read_csv("c:/hello-world-python/machine-learning/cinema.csv")
    except FileNotFoundError:
        print("Erreur : fichier non trouvé.")
        return

    # On sépare nos données :
    # X (Feature) : Ce qu'on connait (La distance)
    # y (Label)   : Ce qu'on veut prédire (Le prix)
    # Attention : Scikit-learn veut X sous forme de table (2D), d'où les doubles crochets [[...]]
    X = df[['distance_km']] 
    y = df['prix_eur']

    print(f"Données chargées : {len(df)} cinémas.")
    print("Exemple de données :")
    print(df.head())

    # --- ÉTAPE 2 : APPRENTISSAGE (TRAINING) ---
    pedagogie_ml(2, "Entraînement du Modèle (Recherche de la droite idéale)")
    print("Le modèle 'LinearRegression' va chercher l'équation : PRIX = a * DISTANCE + b")
    print("Il essaie de minimiser l'erreur totale (la distance entre la droite et les vrais points).")
    
    modele = LinearRegression()
    modele.fit(X, y)
    
    # Récupération des paramètres appris
    coefficient_a = modele.coef_[0] # La pente (a)
    ordonnee_b = modele.intercept_  # L'ordonnée à l'origine (b)
    
    print(f"Modèle appris : PRIX = {coefficient_a:.4f} * DISTANCE + {ordonnee_b:.2f}")
    if coefficient_a > 0:
        print("-> Analyse : Plus c'est loin, plus c'est cher !")
    else:
        print("-> Analyse : Plus c'est loin, moins c'est cher (surprenant...)")

    # --- ÉTAPE 3 : PRÉDICTIONS ET ERREURS ---
    pedagogie_ml(3, "Calcul des Prédictions et des Erreurs")
    # On demande au modèle de prédire le prix pour toutes les distances qu'on connait
    predictions = modele.predict(X)
    
    # On ajoute ces prédictions dans notre tableau pour comparer
    df['prix_predit'] = predictions
    df['erreur'] = df['prix_eur'] - df['prix_predit']
    
    print("Voici un aperçu des erreurs (Réalité vs Modèle) :")
    print(df[['ville', 'distance_km', 'prix_eur', 'prix_predit', 'erreur']].head())

    # --- ÉTAPE 4 : VISUALISATION ---
    pedagogie_ml(4, "Visualisation Graphique : Données vs Modèle")
    
    plt.figure(figsize=(12, 8))
    
    # 1. Les VRAIES données (Points Bleus)
    plt.scatter(df['distance_km'], df['prix_eur'], color='blue', label='Données Réelles (Vérité)', s=100, alpha=0.6)
    
    # 2. La DROITE du modèle (Ligne Rouge)
    plt.plot(df['distance_km'], predictions, color='red', linewidth=3, label='Modèle (Prédiction)')
    
    # 3. Les ERREURS (Lignes pointillées rouges) - C'est très pédagogique !
    # Pour chaque point, on trace une ligne entre le vrai prix et le prix prédit
    print("Génération des lignes d'erreur pour le graphique...")
    for i in range(len(df)):
        x_point = df['distance_km'].iloc[i]
        y_real = df['prix_eur'].iloc[i]
        y_pred = df['prix_predit'].iloc[i]
        
        # On trace une ligne verticale
        plt.plot([x_point, x_point], [y_real, y_pred], color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Mise en forme du graphique
    plt.title(f'Régression Linéaire : Prix vs Distance\nPRIX = {coefficient_a:.2f} * DISTANCE + {ordonnee_b:.2f}', fontsize=16)
    plt.xlabel('Distance depuis St Gervais (km)', fontsize=12)
    plt.ylabel('Prix du Ticket (€)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Ajout d'une annotation pédagogique
    plt.text(X.mean(), y.max(), "Les traits pointillés rouges sont les ERREURS\n(Le modèle essaie de les rendre les plus petits possible)", 
             horizontalalignment='center', color='darkred', style='italic', bbox=dict(facecolor='white', alpha=0.8))

    print("Graphique généré !")
    plt.show()

if __name__ == "__main__":
    main()
