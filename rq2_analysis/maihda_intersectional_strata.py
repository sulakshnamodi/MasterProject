"""
MAIHDA Analysis Pipeline for Research Question 2 (RQ2)

Research Question:
To what extent is variation in literacy proficiency among Norwegian higher education 
graduates attributable to intersectional group membership? Is this variation additive or interactive?

Methodology: 
Multilevel Analysis of Individual Heterogeneity and Discriminatory Accuracy (MAIHDA).
1. Group individuals into intersectional strata (e.g. Track x Gender x SES x Migration x Parent Migration).
2. Fit Model 1 (Null): Individuals nested within strata to compute Variance Partition Coefficient (VPC).
3. Fit Model 2 (Main Effects): Evaluates Proportional Change in Variance (PCV) via additive parameters.
"""

import os
import sys

# Set R environment for pymer4
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.3'
os.environ['R_LIBS_USER'] = r'C:\Users\brind\AppData\Local\R\win-library\4.5'
os.environ['PATH'] = r'C:\Program Files\R\R-4.5.3\bin\x64;' + os.environ['PATH']

import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
from pymer4.models import Lmer

# 1. PATH CONFIGURATION & DATA EXTRACTION
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\maihda'
os.makedirs(outputfolder, exist_ok=True)

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

analysis_df = loaded_data['dataframe'].copy()

# Normalize PIAAC sampling weights
analysis_df['WGT_NORM'] = analysis_df['SPFWT0'] * (len(analysis_df) / analysis_df['SPFWT0'].sum())

# Define analytical categories
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
pv_columns = [f'PVLIT{i}' for i in range(1, 11)]

# Drop rows missing essential anchors or weights
analysis_df = analysis_df.dropna(subset=anchors + pv_columns + ['SPFWT0']).copy()
print(f"Initial analytical cohort size: {len(analysis_df)}")

# 2. INTERSECTIONAL BOUNDARY CREATION
analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

# Filter out very small groupings (< 5 members) for stable MLM outputs
counts = analysis_df['intersectional_id'].value_counts()
valid_ids = counts[counts >= 5].index
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)]
print("Total unique non-empty strata analyzed:", analysis_df['intersectional_id'].nunique())

all_vpcs = []
all_pcvs = []
all_main_effects = []
all_main_ses = []

print("\nRunning MAIHDA models across multiple outcomes...")

# 3. ITERATIVE MODEL FITTING
for pv in pv_columns:
    # Model 1 (Null Model): Estimates pure structural partition bounds
    model_null = Lmer(f"{pv} ~ 1 + (1 | intersectional_id)", data=analysis_df)
    model_null.fit(weights="WGT_NORM", summarize=False)
    var_between_null = float(model_null.ranef_var.iloc[0, 1])
    var_resid_null = getattr(model_null, 'sig2', np.var(model_null.residuals))
    vpc = var_between_null / (var_between_null + var_resid_null)
    all_vpcs.append(vpc)
    
    # Model 2 (Main Effects Model): Captures pure additive explanations
    formula_main = f"{pv} ~ C(GENDER_R) + C(ED_GROUP) + C(PAREDC2) + C(IMPARC2) + C(A2_Q03a_T) + (1 | intersectional_id)"
    model_main = Lmer(formula_main, data=analysis_df)
    model_main.fit(weights="WGT_NORM", summarize=False)
    var_between_main = float(model_main.ranef_var.iloc[0, 1])
    
    # Proportional change evaluates interaction thresholds
    pcv = (var_between_null - var_between_main) / var_between_null
    all_pcvs.append(pcv)
    all_main_effects.append(model_main.coefs['Estimate'])
    all_main_ses.append(model_main.coefs['SE'])
    
# 4. METRIC AGGREGATIONS
avg_vpc = np.mean(all_vpcs)
avg_pcv = np.mean(all_pcvs)

print(f"\n--- INTERSECTIONAL RESULTS ---")
print(f"Average VPC: {avg_vpc:.2%} (Intersectional footprint)")
print(f"Average PCV: {avg_pcv:.2%} (Additive baseline allocation)")

# Rubin's Rules Pooling for Fixed Effects
M = len(pv_columns)
final_coefs = pd.concat(all_main_effects, axis=1).mean(axis=1)
W_fixed = pd.concat(all_main_ses, axis=1).pow(2).mean(axis=1)
B_fixed = pd.concat(all_main_effects, axis=1).var(axis=1)
total_var_fixed = W_fixed + (1 + 1/M) * B_fixed
total_se_fixed = np.sqrt(total_var_fixed)

# P-value extraction
from scipy.stats import norm
z_fixed = final_coefs / total_se_fixed
p_fixed = 2 * (1 - norm.cdf(np.abs(z_fixed)))

fixed_summary = pd.DataFrame({
    'Estimate': final_coefs,
    'Std.Error': total_se_fixed,
    'Z-value': z_fixed,
    'P-value': p_fixed
})

print("\n--- POOLED FIXED EFFECTS (RUBIN'S RULES) ---")
print(fixed_summary.round(4))

# 5. COMPILATION AND GRAPHIC SAVING
analysis_df['PV_AVG'] = analysis_df[pv_columns].mean(axis=1)
plot_df = analysis_df.groupby('intersectional_id').agg({'PV_AVG': 'mean', 'ED_GROUP': 'first'}).reset_index()

def label_strata(row):
    parts = row['intersectional_id'].split('_')
    edu = "Univ" if parts[0] == '0.0' else "Voc"
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses = f"SES-{parts[2][0]}"
    mig = "Native" if parts[4] == '1.0' else "Abroad"
    return f"{edu}-{gen}-{ses}-{mig}"

plot_df['label'] = plot_df.apply(label_strata, axis=1)
plot_df = plot_df.sort_values('PV_AVG', ascending=False)
print("--- TOP 3 COHORTS ---")
print(plot_df[['label', 'PV_AVG']].head(3).to_string(index=False))
print("--- BOTTOM 3 COHORTS ---")
print(plot_df[['label', 'PV_AVG']].tail(3).to_string(index=False))

plt.figure(figsize=(11, 13))
colors = {0.0: "#2b7a5a", 1.0: "#e04c3e"}
sns.barplot(x='PV_AVG', y='label', data=plot_df, hue='ED_GROUP', dodge=False, palette=colors)
plt.axvline(analysis_df['PV_AVG'].mean(), color='black', linestyle='--', label='National Mean')
plt.title('Proficiency Distributions Ranked by MAIHDA Category Blocks', fontsize=14)
plt.xlabel('Literacy Skills (Averaged)')
plt.ylabel('Intersectional Cohort mapping')
plt.xlim(240, 340)
plt.tight_layout()

plot_out = os.path.join(outputfolder, 'rq2_maihda_hierarchy.png')
plt.savefig(plot_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nVisual assets published successfully: {plot_out}")