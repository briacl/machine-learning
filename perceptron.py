
import matplotlib.pyplot as plt
import math

inputs = [2.2, 3.4, 1.6]
input_names = ["x1", "x2", "x3"]

weights = [2.1, 1.3, 6.7]

bias = 3.0

print("--- Calcul du Neurone ---")
print(f"Météo ({inputs[0]}) * Importance ({weights[0]})")
print(f"+ Note ({inputs[1]}) * Importance ({weights[1]})")
print(f"+ Amis ({inputs[2]}) * Importance ({weights[2]})")
print(f"+ Biais ({bias})")

output = (inputs[0] * weights[0]) + \
         (inputs[1] * weights[1]) + \
         (inputs[2] * weights[2]) + \
         bias

print("-" * 20)
print(f"RÉSULTAT DU CALCUL (Potentiel d'activation) : {output}")

active = output > 0
if active:
    print(">>> LE NEURONE S'ACTIVE ! (Je vais au cinéma)")
else:
    print(">>> Le neurone reste inactif. (Je reste chez moi)")

def visualiser_perceptron(inputs, list_weights, bias, output_val, activated):
    print("\nGénération du schéma neuronal...")
    
    bg_color = '#2E2E2E'
    input_color = 'cyan'
    weight_color = 'orange'
    neuron_color = 'magenta'
    text_color = 'white'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    x_input = 1
    x_neuron = 3
    x_output = 4.5
    
    y_inputs = [3, 2, 1]
    y_neuron = 2
    
    for i, y in enumerate(y_inputs):
        circle = plt.Circle((x_input, y), 0.2, facecolor='none', edgecolor=input_color, linewidth=2, zorder=3)
        ax.add_patch(circle)
        
        ax.text(x_input - 0.4, y, f"${input_names[i]}$\n{inputs[i]}", ha='right', va='center', color=input_color)
        
        dx = x_neuron - x_input
        dy = y_neuron - y
        dist = math.sqrt(dx**2 + dy**2)
        
        start_offset = 0.2 + 0.1
        end_offset = 0.5 + 0.1
        
        start_x = x_input + (dx / dist) * start_offset
        start_y = y + (dy / dist) * start_offset
        end_x = x_neuron - (dx / dist) * end_offset
        end_y = y_neuron - (dy / dist) * end_offset

        width = abs(list_weights[i]) 
        ax.plot([start_x, end_x], [start_y, end_y], color=input_color, linewidth=width, zorder=1)
        
        mid_x = (x_input + x_neuron)/2
        mid_y = (y + y_neuron)/2
        ax.text(mid_x, mid_y + 0.15, f"$w_{i+1}$={list_weights[i]}", fontsize=9, color=weight_color, ha='center')

    neuron = plt.Circle((x_neuron, y_neuron), 0.5, facecolor='none', edgecolor=neuron_color, linewidth=3, zorder=3)
    ax.add_patch(neuron)
    ax.text(x_neuron, y_neuron, "$z$", ha='center', va='center', fontsize=20, fontstyle='italic', fontweight='bold', color=neuron_color)
    
    arrow_start_x = x_neuron + 0.5 + 0.1
    ax.arrow(arrow_start_x, y_neuron, 0.5, 0, head_width=0.1, head_length=0.1, fc=neuron_color, ec=neuron_color, linewidth=2)
    
    result_text = "ACTIVÉ !" if activated else "Inactif"
    result_col = 'lime' if activated else 'gray' 
    ax.text(x_output, y_neuron, f"Sortie\n{output_val:.2f}\n{result_text}", 
            ha='left', va='center', fontsize=12, color=result_col, fontweight='bold')

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.axis('off')
    plt.title("Visualisation du Perceptron", fontsize=16, color=text_color)
    
    print("Affichage du graphique.")
    plt.show()

visualiser_perceptron(inputs, weights, bias, output, active)