import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import os

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

analysis_df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
pv_columns = [f'PVLIT{i}' for i in range(1, 11)]

analysis_df = analysis_df.dropna(subset=anchors + pv_columns + ['SPFWT0']).copy()
analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

counts = analysis_df['intersectional_id'].value_counts()
valid_ids = counts[counts >= 5].index
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)].copy()

analysis_df['PV_AVG'] = analysis_df[pv_columns].mean(axis=1)

def label_strata(row):
    parts = row['intersectional_id'].split('_')
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses_map = {'1': 'Low', '2': 'Med', '3': 'High'}
    ses = ses_map.get(parts[2][0], 'Low')
    mig = "Nat" if parts[4] == '1.0' else "For"
    return f"{gen}_{ses}_{mig}"

analysis_df['label'] = analysis_df.apply(label_strata, axis=1)

order_labels = analysis_df.groupby('label')['PV_AVG'].mean().sort_values(ascending=False).index

plt.figure(figsize=(11, 13))
colors = {0.0: "#1b9e77", 1.0: "#d95f02"}
sns.barplot(x='PV_AVG', y='label', data=analysis_df, hue='ED_GROUP', dodge=True, palette=colors, order=order_labels, errorbar=('ci', 95))
plt.axvline(analysis_df['PV_AVG'].mean(), color='black', linestyle='--', label='Grand Mean Proficiency')

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
academic_patch = mpatches.Patch(color='#1b9e77', label='Academic (ISCED 5A/6)')
vocational_patch = mpatches.Patch(color='#d95f02', label='Vocational (ISCED 5B)')
mean_line = mlines.Line2D([], [], color='black', linestyle='--', label='Grand Mean Proficiency')
ci_line = mlines.Line2D([], [], color='#424242', linewidth=2.5, marker='|', markersize=12, label='95% Confidence Interval')

plt.legend(handles=[academic_patch, vocational_patch, mean_line, ci_line], title="", fontsize=14, loc='lower right', frameon=True, facecolor='whitesmoke', edgecolor='gray')

plt.title('Proficiency Distributions Ranked by MAIHDA Category Blocks', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Literacy Skills (Averaged)', fontsize=16, fontweight='bold', labelpad=12)
plt.ylabel('Intersectional Cohort Mapping', fontsize=16, fontweight='bold', labelpad=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(220, 350)
plt.tight_layout()

outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\rq2'
plot_out = os.path.join(outputfolder, 'rq2_maihda_hierarchy.png')
plt.savefig(plot_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nVisual assets published successfully: {plot_out}")
