import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# Load JSON file
filename = "topk_per_cls_accuracies.pkl"  # Change to your file path
with open(filename, "rb") as f:
    data = pkl.load(f)

# Global accuracy values (Scaling to Percentage)
global_accuracies = np.array([0.4968, 0.6695, 0.7607, 0.8113, 0.8473]) * 100  # Convert to %

class_names = [
    "Others",
    "Corn/Sorghum",
    "Alfalfa",
    "Mixed pasture",
    "Misc grain",
    "Rice",
    "Wheat",
    "Tomato",
    "Misc grasses",
    "Native pasture",
    "Cotton",
    "Leafy greens",
    "Onions",
    "Melons",
    "Safflower",
    "Strawberries",
    "Carrots",
    "Sunflower",
    "Potatoes",
    "Bush berries",
    "Sweet potatoes",
    "Sugar beets",
    "Dry beans",
    "Peppers",
]

# Extract class IDs from data keys
classes = list(data.keys())  # List of class IDs (assuming order matches `class_names`)

# Define top-k labels
top_k_labels = ['top_1', 'top_2', 'top_3', 'top_4', 'top_5']

# Convert data to a NumPy array for plotting (Scaling to Percentage)
top_k_accuracies = np.zeros((len(top_k_labels), len(class_names)))
for c in classes:
    for j, k in enumerate(top_k_labels):
        top_k_accuracies[j, int(c)] = data[c][k] * 100  # Convert to percentage

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(20, 7))

x = np.arange(len(class_names))  # Class indices
bar_width = 0.12  # Reduce width slightly for spacing
gap = 0.02  # Small gap between bars


# Colors for different top-k values
colors = ['#D4BEE4', '#BC91B9', '#A4648E', '#8C3663', '#740938']

# Plot bars for each top-k accuracy with a small gap
for i, (label, color) in enumerate(zip(top_k_labels, colors)):
    ax.bar(x + i * (bar_width + gap), top_k_accuracies[i], width=bar_width, label=label, color=color)

# Add global accuracy as horizontal dashed lines
for i, (acc, color) in enumerate(zip(global_accuracies, colors)):
    ax.axhline(y=acc, color=color, linestyle='dashed', linewidth=2, alpha=0.7, label=f'Global {top_k_labels[i]}')

# Formatting the plot
ax.set_xticks(x + (bar_width * 2) + (gap * 2))  # Center x-ticks
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Top-1 to Top-5 Accuracy per Class')
ax.legend(title="Top-K Accuracies")
ax.set_ylim(0, 100)  # Accuracy range

# Save the figure
save_path = "grouped_top_k_accuracies.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"Figure saved successfully at: {save_path}")
