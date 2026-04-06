""" 
RQ1: How do proficiency levels, demographic characteristics, and 
workplace skill use patterns differ between university (ISCED 6–8) and
vocational higher education (ISCED 5) graduates in Norway?
"""

import pandas as pd
import numpy as np
import os, sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# --- DATA LOADING ---
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()

# Ensure we have a normalized weight for the regression models
df['WGT_NORM'] = df['SPFWT0'] * (len(df) / df['SPFWT0'].sum())

# --- 1. DESCRIPTIVE TABLE CALCULATIONS ---
groups = {0.0: 'University', 1.0: 'Vocational'}
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
results = []

for code, label in groups.items():
    df_g = df[df['ED_GROUP'] == code].copy()
    w = df_g['SPFWT0'] 
    
    # Proficiency (Weighted Mean of 10 PVs)
    group_pv_means = [(df_g[pv] * w).sum() / w.sum() for pv in pv_cols]
    weighted_lit_mean = np.mean(group_pv_means)
    
    # Demographics (Weighted Percentages)
    total_w = w.sum()
    pct_female = (df_g[df_g['GENDER_R'] == 2]['SPFWT0'].sum() / total_w) * 100
    pct_high_ses = (df_g[df_g['PAREDC2'] == 3]['SPFWT0'].sum() / total_w) * 100
    pct_foreign_born = (df_g[df_g['A2_Q03a_T'] == 2]['SPFWT0'].sum() / total_w) * 100
    
    # Skill Use (Weighted Mean)
    read_use = (df_g['READWORKC2_WLE_CA_T1'] * w).sum() / total_w
    write_use = (df_g['WRITWORKC2_WLE_CA'] * w).sum() / total_w
    
    results.append({
        'Group': label,
        'Mean Literacy (PV)': round(weighted_lit_mean, 2),
        '% Female': round(pct_female, 1),
        '% High Parent SES': round(pct_high_ses, 1),
        '% Born Abroad': round(pct_foreign_born, 1),
        'Work Read Index': round(read_use, 3),
        'Work Write Index': round(write_use, 3),
        'Sample N': len(df_g)
    })

table_1 = pd.DataFrame(results).set_index('Group')

print("\n" + "="*45)
print("TABLE 1: THE NORWEGIAN HE LANDSCAPE (RQ1)")
print("="*45)
print(table_1.T)

# --- 2. STATISTICAL SIGNIFICANCE TESTING (WLS) ---
# Testing if the difference between groups is statistically significant
test_vars = {
    'PVLIT1': 'Literacy Proficiency',
    'GENDER_R': 'Gender (Female %)',
    'PAREDC2': 'Parental SES',
    'A2_Q03a_T': 'Migration Status',
    'READWORKC2_WLE_CA_T1': 'Work Reading',
    'WRITWORKC2_WLE_CA': 'Work Writing'
}

sig_results = []
for var, name in test_vars.items():
    temp_df = df[[var, 'ED_GROUP', 'WGT_NORM']].dropna()
    # Fit Weighted Least Squares: Characteristic predicted by Education Track
    model = smf.wls(f"{var} ~ ED_GROUP", data=temp_df, weights=temp_df['WGT_NORM']).fit()
    
    sig_results.append({
        'Characteristic': name,
        'Diff (Voc vs Univ)': round(model.params['ED_GROUP'], 3),
        'P-Value': f"{model.pvalues['ED_GROUP']:.4f}",
        'Significant': 'Yes' if model.pvalues['ED_GROUP'] < 0.05 else 'No'
    })

sig_df = pd.DataFrame(sig_results).set_index('Characteristic')
print("\n" + "="*45)
print("SIGNIFICANCE OF GROUP DIFFERENCES")
print("="*45)
print(sig_df)

# --- 3. VISUALIZATION ---
plot_data = table_1.drop(['Sample N', 'Mean Literacy (PV)'], axis=1).reset_index()
plot_data = plot_data.melt(id_vars='Group', var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Value', hue='Group', data=plot_data, palette=['steelblue', 'indianred'])

plt.title('Demographic and Workplace Profiles by Education Track', fontsize=14)
plt.ylabel('Percentage / Index Score')
plt.xlabel('')
plt.xticks(rotation=15)
plt.legend(title='Education Track')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

