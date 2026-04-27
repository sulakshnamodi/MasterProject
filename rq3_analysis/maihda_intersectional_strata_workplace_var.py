import os
import sys

# 1. SET ENVIRONMENT VARIABLES FIRST
# Make sure these paths are 100% correct for your machine
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.3'
os.environ['R_LIBS_USER'] = r'C:\Users\brind\AppData\Local\R\win-library\4.5'
os.environ['PATH'] = r'C:\Program Files\R\R-4.5.3\bin\x64;' + os.environ['PATH']

# 2. NOW IMPORT PYMER4
try:
    from pymer4.models import Lmer
    print("R and Pymer4 initialized successfully!")
except Exception as e:
    print(f"Initialization failed: {e}")

# 3. IMPORT EVERYTHING ELSE
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

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

analysis_df = loaded_data['dataframe'].copy()


# --- PRE-PROCESSING & STRATA CREATION ---
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
# We also drop missing workplace variables for RQ3 consistency
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
analysis_df = analysis_df.dropna(subset=anchors + workplace_vars).copy()

print(f"Analytical sample size for RQ2 is now: {len(analysis_df)}")

# Create Intersectional ID
analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

# Filter small strata (N < 5)
valid_ids = analysis_df['intersectional_id'].value_counts()
valid_ids = valid_ids[valid_ids >= 5].index
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)]

# Normalize Weights
analysis_df['WGT_NORM'] = analysis_df['SPFWT0'] * (len(analysis_df) / analysis_df['SPFWT0'].sum())

# Standardize Workplace Variables (Z-scores)
analysis_df['READ_Z'] = zscore(analysis_df['READWORKC2_WLE_CA_T1'])
analysis_df['WRITE_Z'] = zscore(analysis_df['WRITWORKC2_WLE_CA'])

# --- MODEL DEFINITIONS ---
pv_columns = [f'PVLIT{i}' for i in range(1, 11)]
main_effects = "C(GENDER_R) + C(ED_GROUP) + C(PAREDC2) + C(IMPARC2) + C(A2_Q03a_T)"
rq3_effects = main_effects + " + READ_Z + WRITE_Z"

# Storage for results
results_rq2_vpc = []
results_rq2_pcv = []
results_rq2_coefs = []
results_rq3_coefs = []

print(f"Starting Analysis on {len(analysis_df)} participants across {analysis_df['intersectional_id'].nunique()} strata...")

for pv in pv_columns:
    print(f"Processing {pv}...")
    
    # 1. RQ2: NULL MODEL
    m_null = Lmer(f"{pv} ~ 1 + (1 | intersectional_id)", data=analysis_df)
    m_null.fit(weights="WGT_NORM", summarize=False)
    var_b_null = float(m_null.ranef_var.iloc[0, 1])
    var_r_null = getattr(m_null, 'sig2', np.var(m_null.residuals))
    vpc = var_b_null / (var_b_null + var_r_null)
    results_rq2_vpc.append(vpc)

    # 2. RQ2: MAIN EFFECTS MODEL
    m_main = Lmer(f"{pv} ~ {main_effects} + (1 | intersectional_id)", data=analysis_df)
    m_main.fit(weights="WGT_NORM", summarize=False)
    var_b_main = float(m_main.ranef_var.iloc[0, 1])
    pcv = (var_b_null - var_b_main) / var_b_null
    results_rq2_pcv.append(pcv)
    results_rq2_coefs.append(m_main.coefs)

    # 3. RQ3: WORKPLACE PRACTICE MODEL
    m_rq3 = Lmer(f"{pv} ~ {rq3_effects} + (1 | intersectional_id)", data=analysis_df)
    m_rq3.fit(weights="WGT_NORM", summarize=False)
    results_rq3_coefs.append(m_rq3.coefs)

# --- POOLING & FINAL OUTPUT ---
avg_vpc = np.mean(results_rq2_vpc)
avg_pcv = np.mean(results_rq2_pcv)
pooled_rq2 = pd.concat(results_rq2_coefs).groupby(level=0).mean(numeric_only=True)
pooled_rq3 = pd.concat(results_rq3_coefs).groupby(level=0).mean(numeric_only=True)

print("\n" + "="*30)
print("RESULTS FOR RQ2 (INTERSECTIONALITY)")
print("="*30)
print(f"Average VPC: {avg_vpc:.2%} (Variance due to Strata)")
print(f"Average PCV: {avg_pcv:.2%} (Variance explained by Main Effects)")
print("\nPooled Coefficients (RQ2):")
print(pooled_rq2[['Estimate', 'P-val']])

print("\n" + "="*30)
print("RESULTS FOR RQ3 (WORKPLACE PRACTICE)")
print("="*30)
print("Pooled Coefficients (RQ3):")
# Focus on the 'return' on reading and writing
print(pooled_rq3.loc[['READ_Z', 'WRITE_Z', 'C(ED_GROUP)1'], ['Estimate', 'P-val']])


# 1. Prepare data for Coefficient Plot
# We combine the results for easier comparison
coef_df = pooled_rq3.loc[['READ_Z', 'WRITE_Z', 'C(ED_GROUP)1', 'C(GENDER_R)2'], ['Estimate']].reset_index()
coef_df['index'] = ['Work Reading', 'Work Writing', 'Vocational Gap', 'Gender (Fem)']

# 2. Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Estimate', y='index', data=coef_df, palette='vlag')
plt.axvline(0, color='black', lw=1)
plt.title('Drivers of Literacy Proficiency (RQ3: Workplace Model)', fontsize=14)
plt.xlabel('Point Estimate (Effect on Literacy Score)')
plt.ylabel('Predictor')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# --- VISUALIZATION: THE INTERSECTIONAL HIERARCHY (RQ2) ---

# 1. Calculate the mean of ALL 10 PVs for each person first
analysis_df['PV_AVG'] = analysis_df[pv_columns].mean(axis=1)

# 2. Group by strata and get the mean proficiency
plot_df = analysis_df.groupby('intersectional_id').agg({
    'PV_AVG': 'mean',
    'ED_GROUP': 'first' # Keep 0=Univ, 1=Voc
}).reset_index()

# 3. Create Human-Readable Labels (Optional but highly recommended)
# This replaces the "0.0_1.0_..." strings with actual names for the chart
def label_strata(row):
    parts = row['intersectional_id'].split('_')
    edu = "Univ" if parts[0] == '0.0' else "Voc"
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses = f"SES-{parts[2][0]}" # SES-1, SES-2, SES-3
    mig = "Native" if parts[4] == '1.0' else "Abroad"
    return f"{edu}-{gen}-{ses}-{mig}"

plot_df['label'] = plot_df.apply(label_strata, axis=1)
plot_df = plot_df.sort_values('PV_AVG', ascending=False)
plot_df = plot_df[plot_df['PV_AVG'] > 0]

# 4. Plotting
plt.figure(figsize=(10, 12))
colors = {0.0: "steelblue", 1.0: "indianred"}
palette = [colors[x] for x in plot_df['ED_GROUP']]

sns.barplot(x='PV_AVG', y='label', data=plot_df, hue='ED_GROUP', dodge=False, palette=colors)

plt.axvline(analysis_df['PV_AVG'].mean(), color='black', linestyle='--', label='Norway HE Average')
plt.title('Ranked Proficiency by Intersectional Strata (Norway PIAAC 2023)', fontsize=14)
plt.xlabel('Average Literacy Score (Pooled PVs 1-10)')
plt.ylabel('Intersectional Group')
plt.xlim(240, 340) # Focus on the relevant range
plt.tight_layout()
plt.show()