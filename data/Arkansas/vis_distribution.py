import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def lighten_color(color, amount=0.5):
    """Lighten `color` by mixing with white. amount in [0,1]."""
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(c + (white - c) * amount)

# Load your data
df = pd.read_csv("crop_distribution.csv")

# Sort descending by pct
df = df.sort_values(by="pct", ascending=False)

# Optionally pick top N classes to avoid clutter
top_n = 30
df_plot = df.iloc[:top_n].copy()

# Base strong color for the top (largest) class
base_color = "#1f77b4"  # or whatever you prefer

# Identify the class name of the largest
largest_name = df_plot.iloc[0]["class_name"]

# Generate colors list
colors = []
for idx, row in df_plot.iterrows():
    name = row["class_name"]
    if name == largest_name:
        colors.append(base_color)
    else:
        pos = df_plot.index.get_loc(idx)  # 0-based position
        frac = pos / (len(df_plot) - 1)
        lighten_amt = frac * 0.7  # can tweak max lighten
        colors.append(lighten_color(base_color, amount=lighten_amt))

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df_plot["class_name"], df_plot["pct"], color=colors)

# Use log scale on y-axis
ax.set_yscale("log")

# Rotate x labels
plt.xticks(rotation=90, fontsize=8)
ax.set_xlabel("Class name")
ax.set_ylabel("Percentage (log scale)")
ax.set_title(f"Class distribution (top {top_n}) with annotations")

# Add value labels (pct) above each bar
for bar, pct in zip(bars, df_plot["pct"]):
    # Get bar position and height
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    # Slight offset in “data units” or “points” above bar
    # Here using points offset via text with transform
    ax.text(
        x, 
        y * 1.05,  # 5% above bar height; adjust if too close/far
        f"{pct:.3f}",  # formatting, e.g. 4 decimal places
        ha="center",
        va="bottom",
        fontsize=7,
        rotation=0
    )

plt.tight_layout()

# Save to PNG
output_path = "class_dist_log_pct_annotated.png"
plt.savefig(output_path, dpi=300)
print("Saved to:", output_path)

