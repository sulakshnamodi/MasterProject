"""
MAIHDA Fixed Effects Table Exporter for RQ2.

This script formats the final pooled additive fixed effects generated from the multilevel models.
Outputs an APA 7th Edition compliant table visual.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Data Initialization
# Load mapped estimates computed via Rubin's Rules across plausible outcomes.
data = {
    'Variable': [
        'Sample Baseline (Intercept)',
        'Educational Track (Vocational vs Academic)',
        'Gender (Female vs Male)',
        'Medium SES background vs Low',
        'High SES background vs Low',
        'Parental Migration: One abroad',
        'Parental Migration: Both abroad',
        'Student Migration: Abroad vs Native'
    ],
    'Estimate': [279.80, -34.32, -5.57, 11.14, 18.82, 34.38, 24.65, -18.32],
    'SE': [19.15, 2.96, 2.40, 3.17, 3.30, 21.42, 18.63, 18.90],
    'Z': [14.61, -11.61, -2.33, 3.51, 5.71, 1.61, 1.32, -0.97],
    'P': ['< 0.001 **', '< 0.001 **', '0.020 *', '< 0.001 **', '< 0.001 **', '0.108 ns', '0.186 ns', '0.332 ns']
}

# Step 2: Visualizing APA 7th Edition guidelines
df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.set_xlim(0, 0.95)
ax.set_ylim(-2.5, len(df) + 1.5)
ax.axis('off')

# Border Lines
ax.plot([0.02, 0.93], [len(df) + 0.5, len(df) + 0.5], color='black', lw=1.5)
ax.plot([0.02, 0.93], [len(df) - 0.1, len(df) - 0.1], color='black', lw=1.0)
ax.plot([0.02, 0.93], [-0.3, -0.3], color='black', lw=1.5)

x_coords = [0.03, 0.45, 0.60, 0.72, 0.84]

# Headers
ax.text(x_coords[0], len(df) + 0.1, "Covariate Factor", weight='bold', fontsize=11, fontname='Times New Roman', ha='left')
ax.text(x_coords[1], len(df) + 0.1, "Estimate (B)", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')
ax.text(x_coords[2], len(df) + 0.1, "Std. Error (SE)", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')
ax.text(x_coords[3], len(df) + 0.1, "Z-value", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')
ax.text(x_coords[4], len(df) + 0.1, "P-value", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')

for idx, row in df.iterrows():
    y = len(df) - 1 - idx
    ax.text(x_coords[0], y + 0.1, row['Variable'], fontsize=11, fontname='Times New Roman', color='black', ha='left', va='center')
    ax.text(x_coords[1], y + 0.1, f"{row['Estimate']:.2f}", fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')
    ax.text(x_coords[2], y + 0.1, f"{row['SE']:.2f}", fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')
    ax.text(x_coords[3], y + 0.1, f"{row['Z']:.2f}", fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')
    ax.text(x_coords[4], y + 0.1, row['P'], fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')

# Add VPC and PCV Notes at bottom
ax.text(0.03, -0.8, "Average Variance Partition Coefficient (VPC): 35.69%", fontsize=10, fontname='Times New Roman', style='italic')
ax.text(0.03, -1.4, "Average Proportional Change in Variance (PCV): 99.62%", fontsize=10, fontname='Times New Roman', style='italic')
ax.text(0.03, -2.0, "Note: ** p < 0.01, * p < 0.05, ns = not significant.", fontsize=9, fontname='Times New Roman', style='italic')

plt.suptitle("Table 2.2: Pooled Multilevel Parameter Allocations", fontname='Times New Roman', fontsize=13, weight='bold', y=0.95)

output_dir = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\maihda'
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'rq2_fixed_effects_table.png')

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Fixed effects table published to: {out_path}")
