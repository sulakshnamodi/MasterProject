import pandas as pd
import numpy as np
import pickle
import os

# --- 1. DATA LOADING ---
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()

# --- 2. DEFINE VARIABLES ---
# GENDER_R: 1 = Male, 2 = Female
gender_map = {1.0: 'Male', 2.0: 'Female'}
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
weight_col = 'SPFWT0'

# --- 3. CALCULATE WEIGHTED MEANS ---
gender_results = []

for code, label in gender_map.items():
    # Filter for gender group
    df_g = df[df['GENDER_R'] == code].copy()
    w = df_g[weight_col]
    
    # Calculate weighted mean for each of the 10 Plausible Values
    # Formula: sum(PV * weight) / sum(weights)
    pv_means = [(df_g[pv] * w).sum() / w.sum() for pv in pv_cols]
    
    # Final score is the average of the 10 weighted means
    final_mean = np.mean(pv_means)
    
    gender_results.append({
        'Gender': label,
        'Mean Literacy Score': round(final_mean, 2),
        'Sample N': len(df_g)
    })

# --- 4. DISPLAY RESULTS ---
results_df = pd.DataFrame(gender_results).set_index('Gender')

print("="*40)
print("LITERACY PROFICIENCY BY GENDER (NORWAY)")
print("="*40)
print(results_df)