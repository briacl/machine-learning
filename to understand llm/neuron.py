import matplotlib.pyplot as plt
import math


def calculate_linear_combination(inputs, weights, bias):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
    output += bias
    return output

def first_neuron() :
    inputs = []
    n_inputs = int(input("veuillez saisir le nombre d'entrées : ")) 
    for i in range(n_inputs):
        inputs.append(int(input(f"veuillez saisir l'entrée {i+1} : ")))
    print(f"inputs = {inputs}")

    weights = []
    n_weights = int(input("veuillez saisir le nombre de poids : "))
    for i in range(n_weights):
        weights.append(int(input(f"veuillez saisir le poids {i+1} : ")))
    print(f"weights = {weights}")

    biais = int(input("veuillez saisir le biais : "))
    print(f"biais = {biais}")

    # Use the extracted function
    output = calculate_linear_combination(inputs, weights, biais)
    
    print(f"z (output) = ")
    for i in range(n_inputs):
        print(f"{inputs[i]} * {weights[i]} + ")
    print(f"{biais} = {output}")

    activated = output > 0
    if activated:
        print(">>> LE NEURONE S'ACTIVE !")
    else:
        print(">>> Le neurone reste inactif.")

if __name__ == "__main__":
    first_neuron()