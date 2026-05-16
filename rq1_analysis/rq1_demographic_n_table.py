import matplotlib.pyplot as plt
import pandas as pd
import os

data = [
    ["Gender", "Male", 613, 220, 833],
    ["", "Female", 828, 167, 995],
    ["Migration Status", "Born in Norway", 1200, 346, 1546],
    ["", "Foreign-born", 241, 41, 282],
    ["Parental SES", "Low", 165, 120, 285],
    ["", "Medium", 472, 186, 658],
    ["", "High", 804, 81, 885],
    ["Parental Migration", "Both Native", 1171, 345, 1516],
    ["", "One Foreign", 34, 0, 34],
    ["", "Both Foreign", 236, 42, 278],
]

df = pd.DataFrame(data, columns=['Category', 'Level', 'Academic', 'Vocational', 'Total'])

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 1.0)
ax.set_ylim(-1.5, len(df) + 1.5)
ax.axis('off')

# Borders
ax.plot([0.03, 0.95], [len(df) + 0.5, len(df) + 0.5], color='black', lw=1.5)
ax.plot([0.03, 0.95], [len(df) - 0.1, len(df) - 0.1], color='black', lw=1.0)
ax.plot([0.03, 0.95], [-0.3, -0.3], color='black', lw=1.5)

x_coords = [0.04, 0.28, 0.52, 0.72, 0.88]

# Headers
ax.text(x_coords[0], len(df) + 0.1, "Demographic Category", weight='bold', fontsize=11, fontname='Times New Roman', ha='left')
ax.text(x_coords[1], len(df) + 0.1, "Variable Level", weight='bold', fontsize=11, fontname='Times New Roman', ha='left')
ax.text(x_coords[2], len(df) + 0.1, "Academic (N=1441)", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')
ax.text(x_coords[3], len(df) + 0.1, "Vocational (N=387)", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')
ax.text(x_coords[4], len(df) + 0.1, "Total Sample (N=1828)", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')

for idx, row in df.iterrows():
    y = len(df) - 1 - idx
    ax.text(x_coords[0], y + 0.1, row['Category'], fontsize=11, fontname='Times New Roman', color='black', ha='left', va='center')
    ax.text(x_coords[1], y + 0.1, row['Level'], fontsize=11, fontname='Times New Roman', color='black', ha='left', va='center')
    ax.text(x_coords[2], y + 0.1, f"{row['Academic']:,}", fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')
    ax.text(x_coords[3], y + 0.1, f"{row['Vocational']:,}", fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')
    ax.text(x_coords[4], y + 0.1, f"{row['Total']:,}", fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')

ax.text(0.03, -0.8, "Note: Analytical sample restricted to individuals with complete evaluative datasets.", fontsize=10, fontname='Times New Roman', style='italic')

plt.suptitle("Table 1.1: Analytical Sample Breakdown across Demographic Cohorts", fontname='Times New Roman', fontsize=13, weight='bold', y=0.95)

output_dir = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\rq1'
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'rq1_demographic_n_table.png')

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Demographic table published to: {out_path}")
