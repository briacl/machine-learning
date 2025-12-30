import matplotlib.pyplot as plt
import math

inputs = []
inputs.append(int(input("veuillez saisir une valeur : ")))
inputs.append(int(input("veuillez saisir une valeur : ")))
inputs.append(int(input("veuillez saisir une valeur : ")))
print(f"inputs = {inputs}")

weights = []
weights.append(int(input("veuillez saisir une valeur : ")))
weights.append(int(input("veuillez saisir une valeur : ")))
weights.append(int(input("veuillez saisir une valeur : ")))
print(f"weights = {weights}")

biais = int(input("veuillez saisir une valeur : "))
print(f"biais = {biais}")

output = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + biais
print(f"output = {output}")

activated = output > 0
if activated:
    print(">>> LE NEURONE S'ACTIVE !")
else:
    print(">>> Le neurone reste inactif.")
