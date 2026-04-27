"""
MAIHDA Strata Descriptive Mapping for RQ2.

Groups Norway sample candidates across distinct demographic factors.
Outputs overall APA 7th Edition stratification profiles comfortably.
"""

import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Loading & Anchoring bounds
subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]

df = df.dropna(subset=anchors + pv_cols + ['SPFWT0']).copy()
df['PV_AVG'] = df[pv_cols].mean(axis=1)

df['intersectional_id'] = (
    df['ED_GROUP'].astype(str) + "_" +
    df['GENDER_R'].astype(str) + "_" +
    df['PAREDC2'].astype(str) + "_" +
    df['IMPARC2'].astype(str) + "_" +
    df['A2_Q03a_T'].astype(str)
)

counts = df['intersectional_id'].value_counts()
valid_ids = counts[counts >= 5].index
df_valid = df[df['intersectional_id'].isin(valid_ids)].copy()

def label_strata(row):
    parts = row['intersectional_id'].split('_')
    edu = "Univ" if parts[0] == '0.0' else "Voc"
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses = f"SES-{parts[2][0]}"
    mig = "Native" if parts[4] == '1.0' else "Abroad"
    par_mig = "Both-FB" if parts[3] == '1.0' else ("One-FB" if parts[3] == '2.0' else "Both-NB")
    return f"{edu}-{gen}-{ses}-{par_mig}-{mig}"

# Step 2: Compiling descriptive cohort aggregations
df_valid['Detailed_Label'] = df_valid.apply(label_strata, axis=1)

res = df_valid.groupby('Detailed_Label').agg(
    Mean_Score=('PV_AVG', 'mean'),
    SEM=('PV_AVG', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
    N_Students=('PV_AVG', 'count')
).reset_index()

res = res.sort_values('Mean_Score', ascending=False).reset_index(drop=True)

# -------------------------------------------------------------------------
# Step 3: APA 7th EDITION MATPLOTLIB TABLE EXPORT
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 13))
ax.set_xlim(0, 0.65)
ax.set_ylim(-0.5, len(res) + 1.5)
ax.axis('off')

# Border Lines tightened
ax.plot([0.02, 0.63], [len(res) + 0.5, len(res) + 0.5], color='black', lw=1.5)
ax.plot([0.02, 0.63], [len(res) - 0.1, len(res) - 0.1], color='black', lw=1.0)
ax.plot([0.02, 0.63], [-0.3, -0.3], color='black', lw=1.5)

x_coords = [0.03, 0.08, 0.38, 0.48, 0.58]

# Headers
ax.text(x_coords[0], len(res) + 0.1, "Rank", weight='bold', fontsize=11, fontname='Times New Roman', ha='left')
ax.text(x_coords[1], len(res) + 0.1, "Intersectional Cohort Profile", weight='bold', fontsize=11, fontname='Times New Roman', ha='left')
ax.text(x_coords[2], len(res) + 0.1, "Mean Points", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')
ax.text(x_coords[3], len(res) + 0.1, "SEM (±)", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')
ax.text(x_coords[4], len(res) + 0.1, "N", weight='bold', fontsize=11, fontname='Times New Roman', ha='center')

overall_mean = 279.80
for idx, row in res.iterrows():
    y = len(res) - 1 - idx
    ci_lower = row['Mean_Score'] - 1.96 * row['SEM']
    ci_upper = row['Mean_Score'] + 1.96 * row['SEM']
    sig_flag = "**" if (ci_lower > overall_mean or ci_upper < overall_mean) else ""
    
    ax.text(x_coords[0], y + 0.3, str(idx + 1), fontsize=11, fontname='Times New Roman', color='black', ha='left', va='center')
    ax.text(x_coords[1], y + 0.3, row['Detailed_Label'], fontsize=11, fontname='Times New Roman', color='black', ha='left', va='center')
    ax.text(x_coords[2], y + 0.3, f"{row['Mean_Score']:.2f}{sig_flag}", fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')
    ax.text(x_coords[3], y + 0.3, f"{row['SEM']:.2f}", fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')
    ax.text(x_coords[4], y + 0.3, str(row['N_Students']), fontsize=11, fontname='Times New Roman', color='black', ha='center', va='center')

plt.suptitle("Table 2.1: Intersectional Group Proficiency Ranks", fontname='Times New Roman', fontsize=13, weight='bold', y=0.96)
output_dir = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\maihda'
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'rq2_strata_table.png')

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"APA outcome published to: {out_path}")
