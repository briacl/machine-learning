#!/usr/bin/env python3

"""
Programme de création de matrices - Version améliorée
Basé sur le programme original de matrice.py
Ajout de fonctionnalités pour le machine learning
"""

import numpy as np


def fonction_creation_matrice(m, n, num_matrice):
    """
    Version originale : création manuelle d'une matrice
    m: nombre de lignes
    n: nombre de colonnes
    num_matrice: numéro de la matrice pour l'affichage
    """
    M = []
    for i in range(m):
        ligne = []
        if m > 1:
            print("\n")
        for j in range(n):
            print("M [", 'i', i, "] [", 'j', j, "] = ", end=" ")
            ligne.append(int(input()))
        M.append(ligne)
    MATRICE = np.array(M)
    print("\n > Matrice ", num_matrice, " = \n", MATRICE)
    return MATRICE


def creer_matrice_aleatoire(m, n, num_matrice, min_val=0, max_val=10):
    """
    Création automatique d'une matrice avec des valeurs aléatoires
    Utile pour le machine learning (initialisation de poids, tests, etc.)
    """
    MATRICE = np.random.randint(min_val, max_val, size=(m, n))
    print(f"\n > Matrice {num_matrice} (aléatoire) = \n{MATRICE}")
    return MATRICE


def creer_matrice_zeros(m, n, num_matrice):
    """
    Création d'une matrice remplie de zéros
    Très utilisé en ML pour initialiser des tableaux
    """
    MATRICE = np.zeros((m, n), dtype=int)
    print(f"\n > Matrice {num_matrice} (zéros) = \n{MATRICE}")
    return MATRICE


def creer_matrice_ones(m, n, num_matrice):
    """
    Création d'une matrice remplie de uns
    """
    MATRICE = np.ones((m, n), dtype=int)
    print(f"\n > Matrice {num_matrice} (uns) = \n{MATRICE}")
    return MATRICE


def creer_matrice_identite(n, num_matrice):
    """
    Création d'une matrice identité (diagonale de 1)
    Très importante en algèbre linéaire et ML
    """
    MATRICE = np.eye(n, dtype=int)
    print(f"\n > Matrice {num_matrice} (identité) = \n{MATRICE}")
    return MATRICE


def operations_matricielles(matrices):
    """
    Effectue diverses opérations sur les matrices créées
    """
    if len(matrices) == 0:
        print("\nAucune matrice à traiter.")
        return
    
    print("\n" + "="*50)
    print("OPÉRATIONS MATRICIELLES")
    print("="*50)
    
    # Afficher les dimensions
    for i, mat in enumerate(matrices, 1):
        print(f"\nMatrice {i} : forme {mat.shape}")
    
    # Si on a au moins 2 matrices compatibles, proposer la multiplication
    if len(matrices) >= 2:
        M1, M2 = matrices[0], matrices[1]
        if M1.shape[1] == M2.shape[0]:
            print(f"\n✓ Les matrices 1 et 2 sont compatibles pour la multiplication")
            print(f"  ({M1.shape[0]}x{M1.shape[1]}) × ({M2.shape[0]}x{M2.shape[1]}) = ({M1.shape[0]}x{M2.shape[1]})")
            resultat = M1.dot(M2)
            print(f"\n > Résultat M1 × M2 = \n{resultat}")
        else:
            print(f"\n✗ Les matrices 1 et 2 ne sont pas compatibles pour la multiplication")
            print(f"  M1 a {M1.shape[1]} colonnes, M2 a {M2.shape[0]} lignes")
    
    # Afficher la transposée de la première matrice
    if len(matrices) >= 1:
        print(f"\n > Transposée de la matrice 1 = \n{matrices[0].T}")


def menu_principal():
    """
    Menu interactif pour choisir le mode de création
    """
    print("\n" + "="*50)
    print("PROGRAMME DE CRÉATION DE MATRICES")
    print("="*50)
    print("\nModes de création disponibles :")
    print("1. Manuelle (saisie des valeurs)")
    print("2. Aléatoire (pour tests rapides)")
    print("3. Matrice de zéros")
    print("4. Matrice de uns")
    print("5. Matrice identité")
    print("6. Mode mixte (combinaison)")
    
    choix = input("\nChoisissez un mode (1-6) : ")
    return choix


def mode_mixte():
    """
    Permet de créer plusieurs matrices avec différentes méthodes
    """
    num_matrices = int(input("\nCombien de matrices voulez-vous créer ? : "))
    matrices = []
    
    for i in range(num_matrices):
        print(f"\n--- Matrice {i+1} ---")
        print("1. Manuelle")
        print("2. Aléatoire")
        print("3. Zéros")
        print("4. Uns")
        print("5. Identité")
        
        type_matrice = input("Type de création : ")
        m = int(input("Nombre de lignes : "))
        
        if type_matrice == "5":  # Identité (carrée)
            matrices.append(creer_matrice_identite(m, i+1))
        else:
            n = int(input("Nombre de colonnes : "))
            
            if type_matrice == "1":
                matrices.append(fonction_creation_matrice(m, n, i+1))
            elif type_matrice == "2":
                min_val = int(input("Valeur minimale : "))
                max_val = int(input("Valeur maximale : "))
                matrices.append(creer_matrice_aleatoire(m, n, i+1, min_val, max_val))
            elif type_matrice == "3":
                matrices.append(creer_matrice_zeros(m, n, i+1))
            elif type_matrice == "4":
                matrices.append(creer_matrice_ones(m, n, i+1))
    
    return matrices


def main():
    """
    Fonction principale du programme
    """
    choix = menu_principal()
    matrices = []
    
    if choix == "1":
        # Mode original
        print("\n=== MODE MANUEL ===")
        num_matrices = int(input("\nCombien de matrices voulez-vous créer ? : "))
        for i in range(num_matrices):
            m = int(input(f"\nNombre de lignes pour la matrice {i+1} : "))
            n = int(input(f"Nombre de colonnes pour la matrice {i+1} : "))
            matrices.append(fonction_creation_matrice(m, n, i+1))
    
    elif choix == "2":
        # Mode aléatoire
        print("\n=== MODE ALÉATOIRE ===")
        num_matrices = int(input("\nCombien de matrices voulez-vous créer ? : "))
        for i in range(num_matrices):
            m = int(input(f"\nNombre de lignes pour la matrice {i+1} : "))
            n = int(input(f"Nombre de colonnes pour la matrice {i+1} : "))
            min_val = int(input("Valeur minimale : "))
            max_val = int(input("Valeur maximale : "))
            matrices.append(creer_matrice_aleatoire(m, n, i+1, min_val, max_val))
    
    elif choix == "3":
        # Matrices de zéros
        print("\n=== MATRICES DE ZÉROS ===")
        num_matrices = int(input("\nCombien de matrices voulez-vous créer ? : "))
        for i in range(num_matrices):
            m = int(input(f"\nNombre de lignes pour la matrice {i+1} : "))
            n = int(input(f"Nombre de colonnes pour la matrice {i+1} : "))
            matrices.append(creer_matrice_zeros(m, n, i+1))
    
    elif choix == "4":
        # Matrices de uns
        print("\n=== MATRICES DE UNS ===")
        num_matrices = int(input("\nCombien de matrices voulez-vous créer ? : "))
        for i in range(num_matrices):
            m = int(input(f"\nNombre de lignes pour la matrice {i+1} : "))
            n = int(input(f"Nombre de colonnes pour la matrice {i+1} : "))
            matrices.append(creer_matrice_ones(m, n, i+1))
    
    elif choix == "5":
        # Matrices identité
        print("\n=== MATRICES IDENTITÉ ===")
        num_matrices = int(input("\nCombien de matrices voulez-vous créer ? : "))
        for i in range(num_matrices):
            n = int(input(f"\nDimension (n×n) pour la matrice {i+1} : "))
            matrices.append(creer_matrice_identite(n, i+1))
    
    elif choix == "6":
        # Mode mixte
        matrices = mode_mixte()
    
    # Effectuer des opérations sur les matrices créées
    if matrices:
        operations_matricielles(matrices)
        
        # Sauvegarder les matrices ?
        sauvegarder = input("\n\nVoulez-vous sauvegarder les matrices ? (o/n) : ")
        if sauvegarder.lower() == 'o':
            for i, mat in enumerate(matrices, 1):
                filename = f"matrice_{i}.npy"
                np.save(filename, mat)
                print(f"✓ Matrice {i} sauvegardée dans {filename}")


if __name__ == "__main__":
    main()
