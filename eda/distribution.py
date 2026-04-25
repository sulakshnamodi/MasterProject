import pandas as pd
import pickle
import os

# 1. SETUP
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)
df = loaded_data['dataframe'].copy()

# 2. DEFINE THE TABLE MAPPING
# This dictionary maps your dataframe columns to the table labels and category values
table_config = {
    'EDUCATION TRACK': {'col': 'ED_GROUP', 'vals': {0.0: 'Academic', 1.0: 'Vocational'}},
    'GENDER':          {'col': 'GENDER_R', 'vals': {1.0: 'Male', 2.0: 'Female'}},
    'MIGRATION':       {'col': 'A2_Q03a_T', 'vals': {1.0: 'Native-born', 2.0: 'Foreign-born'}},
    'SES':             {'col': 'PAREDC2',   'vals': {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'}}
}

print(f"Total Analytical Sample (N): {len(df)}")
print("-" * 50)

# 3. CALCULATE AND PRINT
print("--- TABLE 1: DESCRIPTIVES ---")
total_weight = df['SPFWT0'].sum()

for label, info in table_config.items():
    col = info['col']
    for val, cat_name in info['vals'].items():
        subset = df[df[col] == val]
        
        # N = count of rows in the subset
        n_count = len(subset)
        # Weighted percentage
        pct = (subset['SPFWT0'].sum() / total_weight) * 100
        
        # Print matching your requested format
        print(f"{label:<15} | {cat_name:<10} | N={n_count:<5} | %={pct:.2f}")