"""
MAIHDA Variance Decomposition Visualization for RQ2.

Step 1: Environment Configuration and Library Setup
We load required data management and visualization libraries, specifying R environment parameters 
so that Python's pymer4 can interface correctly with R's LME4 software.
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pymer4.models import Lmer

# Set R environment variables for smooth pymer4 interactions on Windows environments.
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.3'
os.environ['PATH'] = r'C:\Program Files\R\R-4.5.3\bin\x64;' + os.environ['PATH']

# Step 2: Path Definition & Preprocessed File Loading
# Point scripts to internal workspace folders ensuring reproducible pipeline outcomes.
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\maihda'
os.makedirs(outputfolder, exist_ok=True)

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

# Step 3: Analytical Cohort Scaling & Cohort Trimming
# Filter missing demographics and normalize overarching weights across targets.
analysis_df = loaded_data['dataframe'].copy()
analysis_df['WGT_NORM'] = analysis_df['SPFWT0'] * (len(analysis_df) / analysis_df['SPFWT0'].sum())

anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
pv_columns = [f'PVLIT{i}' for i in range(1, 11)]

analysis_df = analysis_df.dropna(subset=anchors + pv_columns + ['SPFWT0']).copy()

# Create unique stratum keys for each unique demographic interaction.
analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

# Cull very small sample groups to maintain high standard confidence constraints.
counts = analysis_df['intersectional_id'].value_counts()
valid_ids = counts[counts >= 5].index
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)]

# Step 4: Iterative Multilevel Modeling (Plausible Values 1-10)
# To capture PIAAC measurement error, the pipeline computes individual outcomes across Plausible Values.
all_vpcs = []
all_pcvs = []

print("Processing hierarchical variance parameters...")
for pv in pv_columns:
    # A. Null Model fitting to extract purely geometric baseline differences.
    model_null = Lmer(f"{pv} ~ 1 + (1 | intersectional_id)", data=analysis_df)
    model_null.fit(weights="WGT_NORM", summarize=False)
    var_between_null = float(model_null.ranef_var.iloc[0, 1])
    var_resid_null = getattr(model_null, 'sig2', np.var(model_null.residuals))
    
    # Variance Partition Coefficient evaluates stratum footprint sizes.
    vpc = var_between_null / (var_between_null + var_resid_null)
    all_vpcs.append(vpc)
    
    # B. Main Effects Model mapping additive social indicators safely.
    formula_main = f"{pv} ~ C(GENDER_R) + C(ED_GROUP) + C(PAREDC2) + C(IMPARC2) + C(A2_Q03a_T) + (1 | intersectional_id)"
    model_main = Lmer(formula_main, data=analysis_df)
    model_main.fit(weights="WGT_NORM", summarize=False)
    var_between_main = float(model_main.ranef_var.iloc[0, 1])
    
    # Proportional Change checks standalone structural intersections.
    pcv = (var_between_null - var_between_main) / var_between_null
    all_pcvs.append(pcv)

# Step 5: Pooling Results (Averaging across distributions)
avg_vpc = np.mean(all_vpcs)
sem_vpc = np.std(all_vpcs, ddof=1) / np.sqrt(len(all_vpcs))
avg_pcv = np.mean(all_pcvs)
sem_pcv = np.std(all_pcvs, ddof=1) / np.sqrt(len(all_pcvs))
print(f"--- STATISTICAL METRICS ---")
print(f"VPC: {avg_vpc*100:.4f}% ± {sem_vpc*100:.4f}%")
print(f"PCV: {avg_pcv*100:.4f}% ± {sem_pcv*100:.4f}%")

# Step 6: GRAPHIC GENERATION
# Produces the 100% stacked variance composition bar chart.
fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.cm.BrBG

# Bar A
ax.bar('Bar A:\nTotal Variation\n(Null Model)', avg_vpc * 100, color=cmap(0.9), width=0.6, label='Intersectional Strata Differences', edgecolor='black', linewidth=1, zorder=3)
ax.bar('Bar A:\nTotal Variation\n(Null Model)', (1 - avg_vpc) * 100, bottom=avg_vpc * 100, color=cmap(0.65), width=0.6, label='Individual Differences', edgecolor='black', linewidth=1, zorder=3)

# Bar B
vis_residual = max((1 - avg_pcv) * 100, 3.0)
vis_main = 100 - vis_residual

ax.bar('Bar B:\nStrata Variation\n(Main Effects Model)', vis_main, color='#2b7a5a' , width=0.6, label='Explained by Additive Main Effects', edgecolor='black', linewidth=1, zorder=3)
ax.bar('Bar B:\nStrata Variation\n(Main Effects Model)', vis_residual, bottom=vis_main, color='#e04c3e', width=0.6, label='Unexplained Interaction Residual', edgecolor='black', linewidth=1, zorder=3)

# Styling matching demographics plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

ax.set_ylabel('Percentage of Variance (%)', fontsize=14, fontweight='bold')
ax.set_title('MAIHDA Variance Decomposition Stacked Profile', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 105)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True, facecolor='whitesmoke', edgecolor='gray')

# Annotations
# Bar A
ax.text(0, (avg_vpc * 100) / 2, f"{avg_vpc * 100:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax.text(0, (avg_vpc * 100) + ((1 - avg_vpc) * 100) / 2, f"{(1 - avg_vpc) * 100:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=11)

# Bar B
ax.text(1, vis_main / 2, f"{avg_pcv * 100:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
if (1 - avg_pcv) * 100 < 5:
    ax.text(1, vis_main + (vis_residual / 2), f"{(1 - avg_pcv) * 100:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
else:
    ax.text(1, (avg_pcv * 100) + ((1 - avg_pcv) * 100) / 2, f"{(1 - avg_pcv) * 100:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=11)

plt.tight_layout()
plot_out = os.path.join(outputfolder, 'rq2_maihda_variance_decomp.png')
plt.savefig(plot_out, dpi=300, bbox_inches='tight')
plt.close()

print(f"Success. Output published to: {plot_out}")
