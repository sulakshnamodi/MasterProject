""" RQ3: To what extent does intersectional group membership, based on education type, gender, socioeconomic status and migration background, account for differences in workplace
cognitive literacy skill application among Norwegian adults?  """
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


# RQ1: How do the levels of cognitive literacy skill compare between university and vocationally educated adults at work in Norway?



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

# Storing the results
all_vpcs = []
all_pcvs = []
all_main_effects = []

print("Starting MAIHDA iterations across 10 Plausible Values...")

for pv in pv_columns:
    """  --- MODEL 1: Null Model ---
     Formula: Outcome ~ 1 + (1 | Group), ~ = "Is predicted by..." (literacy score PV is predicted by Grand Mean + the group deviation)
     1 + (1 | intersectional_id), 1: The grand mean, (1 | intersectional_id): the group deviation
     e.g. Grand mean: 250, Group A mean: 230, 1 + (1 | intersectional_id) = 250 + (-20) = 230 """
    
    # Initialising and fitting the model
    model_null = Lmer(f"{pv} ~ 1 + (1 | intersectional_id)", data=analysis_df) 
    model_null.fit(summarize=False)     # this is where actual regression happens

    # Extract variance components for Model 1 (Null)
    # var_between_null: Sigma squared u (Between-strata variance)
    var_between_null = float(model_null.ranef_var.iloc[0, 1])
    
    # var_resid_null: Sigma squared e (Within-strata / Residual variance)
    # Calculate variance from the residuals since .sig2 is missing
    var_resid_null = float(model_null.residuals.var())
    
    # Calculate VPC (Variance Partition Coefficient)
    
    vpc = var_between_null / (var_between_null + var_resid_null)
    all_vpcs.append(vpc)
    
    # --- MODEL 2: Main Effects Model ---
    model_main = Lmer(f"{pv} ~ {main_effects} + (1 | intersectional_id)", data=analysis_df)
    model_main.fit(summarize=False)
    
    var_between_main = float(model_main.ranef_var.iloc[0, 1])
    
    # Calculate PCV (Proportional Change in Variance)
    pcv = (var_between_null - var_between_main) / var_between_null
    all_pcvs.append(pcv)
    
    # Save fixed effects for pooling
    all_main_effects.append(model_main.coefs)

# --- FINAL POOLING ---
avg_vpc = np.mean(all_vpcs)
avg_pcv = np.mean(all_pcvs)

print(f"\n--- INTERSECTIONAL RESULTS ---")
print(f"Average VPC: {avg_vpc:.2%} (Total variance due to intersectional strata)")
print(f"Average PCV: {avg_pcv:.2%} (Proportion of strata variance explained by main effects)")

# Combine fixed effects coefficients (Mean of estimates)
final_coefs = pd.concat(all_main_effects).groupby(level=0).mean(numeric_only=True)
print("\n--- POOLED FIXED EFFECTS (Main Effects) ---")
print(final_coefs[['Estimate', 'P-val']])







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