

#!python3 

# but : écrire un programme de création de matrice

# Bienvenue dans le programme Polynome Resolution by Vector


import numpy as np

def fonction_creation_matrice (m, n, num_matrice) :
    M = []                                                                                       # initialisation de la matrice
    for i in range (m) :                                                                         # pour i de 0 à m
        ligne = []                                                                               # initilaisation de la ligne de la matrice
        if m > 1 :                                                                               # si m est strict sup à 1 alors
            print("\n")                                                                          # afficher un espace entre les lignes
        for j in range(n) :                                                                      # pour j de 0 à n
            print("M [", 'i', i, "] [", 'j', j, "] = ", end=" ")                                 # afficher 
            ligne.append(int(input()))                                                           # on entre la valeur qui s'ajoute dans la case de la ligne
        M.append(ligne)                                                                          # on ajoute la ligne à la matrice
    MATRICE = np.array(M)
    print("\n > Matrice ", num_matrice, " = \n", MATRICE)
    return MATRICE

print("\n Veuillez saisir les variables du vecteur : \n")

# Demandez à l'utilisateur combien de matrices il souhaite créer
num_matrices = int(input("Combien de matrices voulez-vous créer ? : \n"))

# Créez chaque matrice
matrices = []
for i in range(num_matrices):
    m = int(input(f"Combien de lignes voulez-vous que la matrice {i+1} comporte ? : \n"))
    n = int(input(f"Veuillez saisir le degré du polynôme pour la matrice {i+1} : \n"))
    matrices.append(fonction_creation_matrice(m, n, i+1))                                                       # Notez que nous utilisons n1 ici pour assurer la compatibilité des formes

# # Effectuer la multiplication matricielle
# resultat = M1.dot(M2)
# print("\n > Résultat de la multiplication matricielle = \n", resultat)