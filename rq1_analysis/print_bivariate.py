import pickle
import os
import matplotlib.pyplot as plt

subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T', 'IMPARC2']
df = df.dropna(subset=anchors + ['SPFWT0']).copy()

acad_sub = df[df['ED_GROUP'] == 0.0]
acad_w = acad_sub['SPFWT0'].sum()

voc_sub = df[df['ED_GROUP'] == 1.0]
voc_w = voc_sub['SPFWT0'].sum()

data = []

def get_pcts(col_name, val_map):
    row_block = []
    first = True
    for val, lbl in val_map.items():
        acad_pct = acad_sub[acad_sub[col_name] == val]['SPFWT0'].sum() / acad_w * 100
        voc_pct = voc_sub[voc_sub[col_name] == val]['SPFWT0'].sum() / voc_w * 100
        dim_lbl = col_name if first else ""
        row_block.append([dim_lbl, lbl, f"{acad_pct:.2f}%", f"{voc_pct:.2f}%"])
        first = False
    return row_block

data.extend(get_pcts("GENDER_R", {1.0: "Male", 2.0: "Female"}))
data.extend(get_pcts("PAREDC2", {1.0: "Low", 2.0: "Medium", 3.0: "High"}))
data.extend(get_pcts("A2_Q03a_T", {1.0: "Native", 2.0: "Foreign"}))

for row in data:
    if row[0] == "GENDER_R": row[0] = "Gender"
    elif row[0] == "PAREDC2": row[0] = "Parental SES"
    elif row[0] == "A2_Q03a_T": row[0] = "Migration"

# -------------------------------------------------------------------------
# APA 7th EDITION - TIGHTER COLUMN MAPPING
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.set_xlim(0, 0.75)
ax.set_ylim(-0.8, 6.8)
ax.axis('off')

# Pulled columns closer together
y_pos = [5.0, 4.3, 3.3, 2.6, 1.9, 0.9, 0.2]
x_coords = [0.05, 0.22, 0.44, 0.62]

# Border Lines tightened
ax.plot([0.03, 0.72], [6.0, 6.0], color='black', lw=1.5)
ax.plot([0.03, 0.72], [5.4, 5.4], color='black', lw=1.0)
ax.plot([0.03, 0.72], [-0.3, -0.3], color='black', lw=1.5)

# Headers
ax.text(x_coords[0], 5.6, "Variable", weight='bold', fontsize=11, fontname='Times New Roman', ha='left')
ax.text(x_coords[1], 5.6, "Category", weight='bold', fontsize=11, fontname='Times New Roman', ha='left')
ax.text(x_coords[2], 5.6, "Academic (%)", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')
ax.text(x_coords[3], 5.6, "Vocational (%)", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')

for idx, row in enumerate(data):
    y = y_pos[idx]
    ax.text(x_coords[0], y, row[0], fontsize=11, fontname='Times New Roman', weight='bold' if row[0] else 'normal', color='black', ha='left', va='center')
    ax.text(x_coords[1], y, row[1], fontsize=11, fontname='Times New Roman', color='black', ha='left', va='center')
    ax.text(x_coords[2], y, row[2], fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')
    ax.text(x_coords[3], y, row[3], fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')

plt.suptitle("Table 1.1: Baseline Demographic Splits", fontname='Times New Roman', fontsize=13, weight='bold', y=0.96)
output_dir = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\rq1'
out_path = os.path.join(output_dir, 'rq1_bivariate_table.png')

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Tighter Columns mapped.")
