# RQ2: To what extent is variation in literacy and numeracy proficiency among Norwegian higher education graduates attributable to intersectional group membership (defined by education type, gender, socioeconomic background, and migration status), and is this variation primarily additive or interactive?
import os
os.environ['RPY2_CFFI_MODE'] = 'ABI'
os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.5.3'
os.environ['PATH'] = 'C:\\Program Files\\R\\R-4.5.3\\bin\\x64' + os.pathsep + os.environ['PATH']
import sys
import pandas as pd
import numpy as np
import pickle 
from scipy.stats import norm  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pymer4.models import Lmer
import statsmodels.formula.api as smf

# Setting of subdataset file
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\regression'

# load the pickle file back
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)


subdataset_df = loaded_data['dataframe']

# Drop rows with missing values in your intersectional anchors
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
analysis_df = subdataset_df.dropna(subset=anchors).copy()

for col in analysis_df.columns:
    if col not in anchors:
        continue
    unique_vals = analysis_df[col].unique()
    print(f"Column: {col} | Unique Values: {unique_vals}")


metadata = loaded_data['metadata'] # dictionary from the codebook

# Create the Intersectional Strata ID
analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

# Remove very small strata
counts = analysis_df['intersectional_id'].value_counts()
valid_ids = counts[counts >= 5].index
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)]

print("Number of strata after removing small groups:",
      analysis_df['intersectional_id'].nunique())

print(f"Number of unique intersectional strata: {analysis_df['intersectional_id'].nunique()}")

# This counts how many people are in every unique combination
counts = analysis_df.groupby(['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T' ]).size()
print(counts)
print(f"Total number of non-empty strata: {len(counts)}")

# Calculate number of empty strata
total_theoretical_strata = 1
for col in anchors:
    total_theoretical_strata *= analysis_df[col].nunique()

print(f"Total theoretical strata: {total_theoretical_strata}")
print(f"Number of empty strata: {total_theoretical_strata - len(counts)}")


# Iterate through Plausible Values (PVs)
# Loop through each of the 10 Literacy scores (PVLIT1 to PVLIT10)
pv_columns = [f'PVLIT{i}' for i in range(1, 11)]

""" applying the logic behind Model 2, that is the Main Effects of adding all the coeffcient one by one after a standard regression
 - basically what score we would expect (main effect/additive effect)
 - literacy score of person x = 25, 19 = main effect, +6 = intersectional effect

 - 19 (+15 (university education), +2(female) +5(parents education: university), +3 (both parents foreign born)-6 (not born in Norway)) """

main_effects = "C(GENDER_R) + C(ED_GROUP) + C(PAREDC2) + C(IMPARC2) + C(A2_Q03a_T)"    # C(...) is used to treat the varaibles as categorial (string) as we are using pymer4

# Normalize the weights (Recommended for multilevel models)
# This ensures the sum of weights equals the number of rows
analysis_df['WGT_NORM'] = analysis_df['SPFWT0'] * (len(analysis_df) / analysis_df['SPFWT0'].sum())

# Storing the results
all_vpcs = []
all_pcvs = []
all_main_effects = []

print("Starting MAIHDA iterations across 10 Plausible Values...")

for pv in pv_columns:
    # --- MODEL 1: Null Model (Weighted) ---
    model_null = Lmer(f"{pv} ~ 1 + (1 | intersectional_id)", data=analysis_df) 
    
    # Pass the normalized weight column to the fit method
    model_null.fit(weights="WGT_NORM", summarize=False) 

    # Robust Variance Extraction
    # .ranef_var is a DataFrame: the first row is usually your random effect
    var_between_null = float(model_null.ranef_var.iloc[0, 1]) 
    
    # Residual variance is often stored in .sig2 in pymer4
    # If .sig2 isn't available, we use the squared standard deviation of residuals
    var_resid_null = getattr(model_null, 'sig2', np.var(model_null.residuals))
    
    # Calculate VPC
    vpc = var_between_null / (var_between_null + var_resid_null)
    all_vpcs.append(vpc)
    
    # --- MODEL 2: Main Effects Model (Weighted) ---
    # We use the categorical wrapper C() for R-style factors
    model_main = Lmer(f"{pv} ~ C(GENDER_R) + C(ED_GROUP) + C(PAREDC2) + C(IMPARC2) + C(A2_Q03a_T) + (1 | intersectional_id)", data=analysis_df)
    
    # Pass the normalized weight column to the fit method
    model_main.fit(weights="WGT_NORM", summarize=False)
    
    var_between_main = float(model_main.ranef_var.iloc[0, 1])
    
    # Calculate PCV (Proportional Change in Variance)
    # If var_between_main is very close to 0, PCV will approach 1.0 (100%)
    pcv = (var_between_null - var_between_main) / var_between_null
    all_pcvs.append(pcv)
    
    # Save coefficients for pooling
    all_main_effects.append(model_main.coefs)

# --- FINAL POOLING ---
avg_vpc = np.mean(all_vpcs)
avg_pcv = np.mean(all_pcvs)

print(f"\n--- WEIGHTED INTERSECTIONAL RESULTS ---")
print(f"Average VPC: {avg_vpc:.2%} (Total variance due to intersectional position)")
print(f"Average PCV: {avg_pcv:.2%} (Proportion explained by additive main effects)")

# Combine fixed effects coefficients
final_coefs = pd.concat(all_main_effects).groupby(level=0).mean(numeric_only=True)
print("\n--- POOLED FIXED EFFECTS (Weighted) ---")
print(final_coefs[['Estimate', 'P-val']])


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



# Model 1 (The Null/Baseline Model)
    # - Calculate VPC (Variance Partition Coefficient)
# Model 2 (The Main Effects Model)
    # - Calculate PCV (Portional Change in Variamce)
# Pool Results: Use Rubin's Rules to average the result across the 10 PVs


""" Note:
What is a "Good" VPC for your study?
In social science and MAIHDA:

1% to 5%: Small intersectional effect (Most differences are individual).

5% to 20%: Moderate intersectional effect (The "pockets" of society matter quite a bit).

> 20%: Large intersectional effect (Your social identity is a very strong predictor of your skills). """