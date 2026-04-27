"""
MAIHDA Caterpillar Plot Analysis Pipeline for Research Question 2 (RQ2)

Generates a ranked random effects plot (residuals) representing strata deviations 
from the purely additive baseline using Rubin's Rules across Plausible Values.
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
import rpy2.robjects as ro
from scipy.stats import norm

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

all_blups = []
all_vars = []

print("\nFitting models and extracting stratum residuals...")

# 3. ITERATIVE MODEL FITTING WITH LMER (R)
for pv in pv_columns:
    print(f"Processing {pv}...")
    formula_main = f"{pv} ~ C(GENDER_R) + C(ED_GROUP) + C(PAREDC2) + C(IMPARC2) + C(A2_Q03a_T) + (1 | intersectional_id)"
    model_main = Lmer(formula_main, data=analysis_df)
    model_main.fit(weights="WGT_NORM", summarize=False)
    
    # Extract random intercepts & conditional variances using R bridges
    ro.globalenv['m'] = model_main.model_obj
    ro.r('library(lme4)')
    ro.r('re <- ranef(m, condVar=TRUE)')
    ro.r('blups <- re[[1]]')
    ro.r('vars <- attr(blups, "postVar")')
    ro.r('blups_vec <- blups[,1]')
    ro.r('strata_names <- rownames(blups)')
    
    blups_vector = np.array(ro.r('blups_vec'))
    vars_vector = np.array(ro.r('as.vector(vars)'))
    strata_names = list(ro.r('strata_names'))
    
    all_blups.append(pd.Series(blups_vector, index=strata_names))
    all_vars.append(pd.Series(vars_vector, index=strata_names))

# 4. RUBIN'S RULES POOLING
M = len(pv_columns)
pooled_blups = pd.concat(all_blups, axis=1).mean(axis=1)
W = pd.concat(all_vars, axis=1).mean(axis=1)
B = pd.concat(all_blups, axis=1).var(axis=1)
total_var = W + (1 + 1/M) * B
total_se = np.sqrt(total_var)

ci_lower = pooled_blups - 1.96 * total_se
ci_upper = pooled_blups + 1.96 * total_se

# 5. DATA COMPILATION & SORTING
caterpillar_df = pd.DataFrame({
    'strata': pooled_blups.index,
    'blup': pooled_blups.values,
    'se': total_se.values,
    'ci_lower': ci_lower.values,
    'ci_upper': ci_upper.values
})

# Translate strata labels
def label_strata(strata_id):
    parts = strata_id.split('_')
    edu = "Univ" if parts[0] == '0.0' else "Voc"
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses = f"SES-{parts[2][0]}"
    
    if parts[3] == '3.0':
        par = "ParNat"
    elif parts[3] == '2.0':
        par = "ParMix"
    else:
        par = "ParAbr"
        
    mig = "Native" if parts[4] == '1.0' else "Abroad"
    return f"{edu}-{gen}-{ses}-{par}-{mig}"

caterpillar_df['label'] = caterpillar_df['strata'].apply(label_strata)
caterpillar_df = caterpillar_df.sort_values('blup').reset_index(drop=True)

# Calculate p-values and significance stars via Z-score
caterpillar_df['z_score'] = caterpillar_df['blup'] / caterpillar_df['se']
caterpillar_df['p_value'] = 2 * (1 - norm.cdf(np.abs(caterpillar_df['z_score'])))

def get_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

caterpillar_df['stars'] = caterpillar_df['p_value'].apply(get_stars)

# Step 5: Flag significant independent interactions
caterpillar_df['significant'] = caterpillar_df['stars'] != ""

# Step 6: GRAPHIC GENERATION
plt.figure(figsize=(14, 10))
y_pos = np.arange(len(caterpillar_df))

# Posh color palettes
colors = ['#e74c3c' if sig else '#2c3e50' for sig in caterpillar_df['significant']]

plt.errorbar(
    caterpillar_df['blup'], y_pos, 
    xerr=1.96 * caterpillar_df['se'], 
    fmt='none', ecolor='#bdc3c7', elinewidth=2, capsize=4
)

for i, row in caterpillar_df.iterrows():
    plt.plot(row['blup'], y_pos[i], 'o', color=colors[i], markersize=9, markeredgecolor='white', markeredgewidth=1)
    if row['stars'] != "":
        x_txt = row['ci_upper'] + 0.2
        plt.text(x_txt, y_pos[i], row['stars'], va='center', ha='left', color='#e74c3c', fontsize=12, fontweight='bold', fontname='Times New Roman')

plt.axvline(0, color='#333333', linestyle='--', linewidth=1.5)

# Create custom legend handles including CI and Significance Key
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#333333', lw=1.5, linestyle='--', label='Additive Baseline (0)'),
    Line2D([0], [0], color='#bdc3c7', lw=2, label='95% Confidence Interval (CI)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=9, markeredgecolor='white', label='Significant Interaction'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c3e50', markersize=9, markeredgecolor='white', label='Non-Significant (Purely Additive)')
]
plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0., ncol=2, fontsize=11, frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

plt.yticks(y_pos, caterpillar_df['label'], fontsize=11, fontname='Times New Roman')
plt.xlabel('Intersectional Deviation (Random Effect Residuals)', fontsize=12, fontname='Times New Roman', labelpad=12)
plt.ylabel('Intersectional Cohort Profiles', fontsize=12, fontname='Times New Roman', labelpad=12)
plt.title('MAIHDA Caterpillar Plot: Testing for Intersectional Interactions', fontsize=14, weight='bold', fontname='Times New Roman', pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.5)

caterpillar_out = os.path.join(outputfolder, 'rq2_maihda_caterpillar.csv')
caterpillar_df.to_csv(caterpillar_out, index=False)
print(f"CSV metrics saved to: {caterpillar_out}")

plot_out = os.path.join(outputfolder, 'rq2_maihda_caterpillar.png')
plt.savefig(plot_out, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nThe MAIHDA Caterpillar Analysis is complete. Figure published to: {plot_out}")
