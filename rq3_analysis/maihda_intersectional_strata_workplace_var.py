import os
import sys

# 1. SET ENVIRONMENT VARIABLES FIRST
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.3'
os.environ['R_LIBS_USER'] = r'C:\Users\brind\AppData\Local\R\win-library\4.5'
os.environ['PATH'] = r'C:\Program Files\R\R-4.5.3\bin\x64;' + os.environ['PATH']

from pymer4.models import Lmer
import pandas as pd
import numpy as np
import pickle
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

# --- DATA LOADING ---
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)

outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\rq3'
os.makedirs(outputfolder, exist_ok=True)

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

analysis_df = loaded_data['dataframe'].copy()

# --- PRE-PROCESSING & STRATA CREATION ---
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
analysis_df = analysis_df.dropna(subset=anchors + workplace_vars + ['SPFWT0']).copy()

analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

valid_ids = analysis_df['intersectional_id'].value_counts()
valid_ids = valid_ids[valid_ids >= 5].index
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)]

analysis_df['WGT_NORM'] = analysis_df['SPFWT0'] * (len(analysis_df) / analysis_df['SPFWT0'].sum())
analysis_df['READ_Z'] = zscore(analysis_df['READWORKC2_WLE_CA_T1'])
analysis_df['WRITE_Z'] = zscore(analysis_df['WRITWORKC2_WLE_CA'])

# --- MODEL DEFINITIONS ---
pv_columns = [f'PVLIT{i}' for i in range(1, 11)]
main_effects = "C(GENDER_R) + C(ED_GROUP) + C(PAREDC2) + C(IMPARC2) + C(A2_Q03a_T)"
rq3_effects = main_effects + " + READ_Z + WRITE_Z"

results_rq2_vpc = []
results_rq2_pcv = []
results_rq2_coefs = []
results_rq3_pcv = []
results_rq3_coefs = []

print(f"Starting Analysis on {len(analysis_df)} participants across {analysis_df['intersectional_id'].nunique()} strata...")

# Fit Plausible Values
for pv in pv_columns:
    print(f"Processing {pv}...")
    
    m_null = Lmer(f"{pv} ~ 1 + (1 | intersectional_id)", data=analysis_df)
    m_null.fit(weights="WGT_NORM", summarize=False)
    var_b_null = float(m_null.ranef_var.iloc[0, 1])
    var_r_null = getattr(m_null, 'sig2', np.var(m_null.residuals))
    vpc = var_b_null / (var_b_null + var_r_null)
    results_rq2_vpc.append(vpc)

    m_main = Lmer(f"{pv} ~ {main_effects} + (1 | intersectional_id)", data=analysis_df)
    m_main.fit(weights="WGT_NORM", summarize=False)
    var_b_main = float(m_main.ranef_var.iloc[0, 1])
    pcv = (var_b_null - var_b_main) / var_b_null
    results_rq2_pcv.append(pcv)
    results_rq2_coefs.append(m_main.coefs)

    # RQ3 random slopes model
    m_rq3 = Lmer(f"{pv} ~ {rq3_effects} + (1 | intersectional_id) + (0 + READ_Z | intersectional_id)", data=analysis_df)
    m_rq3.fit(weights="WGT_NORM", summarize=False)
    var_b_rq3 = float(m_rq3.ranef_var.iloc[0, 1])
    pcv_rq3 = (var_b_null - var_b_rq3) / var_b_null
    results_rq3_pcv.append(pcv_rq3)
    results_rq3_coefs.append(m_rq3.coefs)

avg_vpc = np.mean(results_rq2_vpc)
avg_pcv = np.mean(results_rq2_pcv)
avg_pcv_rq3 = np.mean(results_rq3_pcv)
pooled_rq2 = pd.concat(results_rq2_coefs).groupby(level=0).mean(numeric_only=True)
pooled_rq3 = pd.concat(results_rq3_coefs).groupby(level=0).mean(numeric_only=True)

print("\n" + "="*30)
print("RESULTS FOR RQ3 (WORKPLACE PRACTICE)")
print("="*30)
print(f"Average VPC: {avg_vpc:.2%}")
print(f"Total Intersectional PCV with Workplace practices: {avg_pcv_rq3:.2%}")
print(f"Additional Intersectional Variance Explained by Skills: {avg_pcv_rq3 - avg_pcv:.2%}")
print("\nPooled Coefficients (RQ3):")
print(pooled_rq3.loc[['READ_Z', 'WRITE_Z', 'C(ED_GROUP)1'], ['Estimate', 'P-val']])

# --- VISUALIZATION 1: Forest Plot (Option 3) ---
coef_df = pooled_rq3.loc[['READ_Z', 'WRITE_Z', 'C(ED_GROUP)1', 'C(GENDER_R)2'], ['Estimate']].reset_index()

def get_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

coef_df['index'] = [
    f"Work Reading{get_stars(pooled_rq3.loc['READ_Z', 'P-val'])}",
    f"Work Writing{get_stars(pooled_rq3.loc['WRITE_Z', 'P-val'])}",
    f"Vocational Gap{get_stars(pooled_rq3.loc['C(ED_GROUP)1', 'P-val'])}",
    f"Gender (Fem){get_stars(pooled_rq3.loc['C(GENDER_R)2', 'P-val'])}"
]

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Estimate', y='index', data=coef_df, palette='BrBG', edgecolor='black', linewidth=1)
plt.axvline(0, color='black', lw=2)

ax.set_title('Drivers of Literacy Proficiency (RQ3: Workplace Model)', fontsize=14, pad=15)
ax.set_xlabel('Point Estimate (Effect on Literacy Score)', fontsize=14, labelpad=15)
ax.set_ylabel('Predictor', fontsize=14, labelpad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.figtext(0.05, 0.02, "* p < 0.05, ** p < 0.01, *** p < 0.001", fontsize=10, style='italic')

forest_out = os.path.join(outputfolder, 'rq3_predictors_forest.png')
plt.savefig(forest_out, dpi=300, bbox_inches='tight')
plt.close()

# --- VISUALIZATION 2: Random Slopes (Option 1) ---
fixef = pooled_rq3['Estimate']
# Extracting ranef from the LAST evaluated plausible outcome safely
ranef = m_rq3.ranef

x_vals = np.linspace(-2, 2, 100)

def label_strata(strat_id):
    parts = strat_id.split('_')
    edu = "Univ" if parts[0] == '0.0' else "Voc"
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses = f"SES-{parts[2][0]}"
    mig = "Native" if parts[4] == '1.0' else "Abroad"
    return f"{edu}-{gen}-{ses}-{mig}"

strata_sizes = analysis_df['intersectional_id'].value_counts()

plt.figure(figsize=(12, 8))

for strat in ranef.index:
    try:
        strat_int = ranef.loc[strat, '(Intercept)']
        strat_slope = ranef.loc[strat, 'READ_Z']
    except:
        strat_int = ranef.loc[strat].iloc[0]
        strat_slope = ranef.loc[strat].iloc[1]
        
    total_slope = fixef.loc['READ_Z'] + strat_slope
    y_vals = (fixef.loc['(Intercept)'] + strat_int) + total_slope * x_vals
    
    lbl = label_strata(strat)
    color = '#2b7a5a' if 'Univ' in lbl else '#e04c3e'
    
    # Weighted line logic: thicker/brighter for larger sample strata
    n_size = strata_sizes.get(strat, 5)
    lw_weight = 1.0 + (n_size / strata_sizes.max()) * 4.0
    alpha_weight = 0.2 + (n_size / strata_sizes.max()) * 0.7
    
    plt.plot(x_vals, y_vals, color=color, alpha=alpha_weight, lw=lw_weight, label=lbl if lbl not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Intersectional Returns on Workplace Reading Skills (Random Slopes)", fontsize=14, weight='bold')
plt.xlabel("Workplace Reading Practice Frequency (Z-Score)")
plt.ylabel("Literacy Proficiency Outcomes (Predictions)")
plt.grid(True, linestyle='--', alpha=0.5)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2b7a5a', lw=4, label='University Cohorts'),
    Line2D([0], [0], color='#e04c3e', lw=4, label='Vocational Cohorts'),
    Line2D([0], [0], color='gray', lw=4, alpha=0.9, label='Large Sample Profile'),
    Line2D([0], [0], color='gray', lw=1.5, alpha=0.3, label='Small Sample Profile')
]
plt.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd', title="Cohort & Reliability")

slope_out = os.path.join(outputfolder, 'rq3_random_slopes.png')
plt.savefig(slope_out, dpi=300, bbox_inches='tight')
plt.close()

# --- VISUALIZATION 3: Workplace Skill Access Heatmap (Option 2) ---
# Group access frequencies by individual strata capabilities
access_df = analysis_df.groupby('intersectional_id').agg({
    'READ_Z': 'mean',
    'WRITE_Z': 'mean',
    'ED_GROUP': 'first'
}).reset_index()

access_df['label'] = access_df['intersectional_id'].apply(label_strata)
access_df = access_df.sort_values('READ_Z', ascending=False)

plt.figure(figsize=(10, 12))
ax = sns.heatmap(
    access_df.set_index('label')[['READ_Z', 'WRITE_Z']], 
    cmap='BrBG', 
    center=0,
    annot=True, 
    fmt=".2f", 
    linewidths=1, 
    linecolor='black',
    annot_kws={"size": 12},
    cbar_kws={'label': 'Standardized Access frequency (Z-Score)'}
)

ax.set_title("Differential Access to Workplace Cognitive Skills across Strata", fontsize=14, pad=15)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
ax.set_xlabel("Workplace Practice Modalities", fontsize=14, labelpad=15)
ax.set_ylabel("Intersectional Stratum Profiles", fontsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.tight_layout()

heatmap_out = os.path.join(outputfolder, 'rq3_strata_skill_heatmap.png')
plt.savefig(heatmap_out, dpi=300, bbox_inches='tight')
plt.close()

print(f"All analyses outputs (including Option 2 access benchmarks) published to {outputfolder}")