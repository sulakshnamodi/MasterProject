import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap

# -----------------------------------------------------------------------------
# 1. DATA CONFIGURATION
# -----------------------------------------------------------------------------
output_dir = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\rq1_analysis'
os.makedirs(output_dir, exist_ok=True)

labels = ['Academic (ISCED 5A/6)', 'Vocational (ISCED 5B)']
sizes = [73.91, 26.09]
counts = [1441, 387]
colors = ['#1b9e77', '#d95f02']

# -----------------------------------------------------------------------------
# OPTION 1: THE PREMIUM DONUT CHART
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

wedges, texts, autotexts = ax.pie(
    sizes, 
    labels=labels, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=colors, 
    pctdistance=0.75,
    wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2),
    textprops={'fontsize': 14, 'weight': 'bold'}
)

# Central circle for baseline text
ax.text(0, 0, f'Total N\n{sum(counts):,}', ha='center', va='center', fontsize=16, weight='bold')

plt.setp(autotexts, size=13, weight="bold", color="white")
ax.set_title("Track Allocation Distribution (Donut)", fontsize=16, pad=20, weight='bold')

plot1_path = os.path.join(output_dir, 'fancy_track_breakdown_donut.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# OPTION 2: THE WAFFLE CHART (MANUAL GRID)
# -----------------------------------------------------------------------------
grid = np.zeros((10, 10))
# 74 blocks for Academic, 26 for Vocational
grid.flat[:74] = 0
grid.flat[74:] = 1

fig, ax = plt.subplots(figsize=(7, 7))
cmap = ListedColormap(colors)

# Plot grid blocks
ax.imshow(grid, cmap=cmap, aspect='equal')

# Add discrete grid separation lines
ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=3)

# Format constraints
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)

# Dynamic legend markers
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=colors[0], label=f'{labels[0]} ({sizes[0]}%)'),
    mpatches.Patch(color=colors[1], label=f'{labels[1]} ({sizes[1]}%)')
]
ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
ax.set_title("Track Allocation Distribution (Waffle Chart)", fontsize=16, pad=20, weight='bold')

plot2_path = os.path.join(output_dir, 'fancy_track_breakdown_waffle.png')
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# OPTION 3: FLOATING STACKED PERCENTAGE BAR
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3))

# Draw bars
ax.barh([0], [sizes[0]], color=colors[0], height=0.6, label=labels[0], edgecolor='white', linewidth=2)
ax.barh([0], [sizes[1]], left=[sizes[0]], color=colors[1], height=0.6, label=labels[1], edgecolor='white', linewidth=2)

# Labels inside segments
ax.text(sizes[0]/2, 0, f'{sizes[0]}%\n(N={counts[0]})', ha='center', va='center', color='white', fontsize=14, weight='bold')
ax.text(sizes[0] + sizes[1]/2, 0, f'{sizes[1]}%\n(N={counts[1]})', ha='center', va='center', color='white', fontsize=14, weight='bold')

# Styling limits
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 0.5)
ax.axis('off')

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=12)
ax.set_title("Track Allocation Distribution (Stacked Proportions)", fontsize=16, pad=20, weight='bold')

plot3_path = os.path.join(output_dir, 'fancy_track_breakdown_stacked.png')
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
plt.close()

print("Generated all three visualization variants successfully.")
