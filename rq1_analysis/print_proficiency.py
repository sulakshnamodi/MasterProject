import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle
import os

subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0', 'AVG_LIT']).copy()

mapping_config = {
    'ED_GROUP': {0.0: 'Academic', 1.0: 'Vocational'},
    'GENDER_R': {1.0: 'Male', 2.0: 'Female'},
    'PAREDC2': {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'}
}

for col in ['GENDER_R', 'PAREDC2']:
    print(f"\n=== {col} ===")
    for cat_val, cat_lbl in mapping_config[col].items():
        for t_val, t_lbl in mapping_config['ED_GROUP'].items():
            sub = df[(df[col] == cat_val) & (df['ED_GROUP'] == t_val)]
            model = smf.wls("AVG_LIT ~ 1", data=sub, weights=sub['SPFWT0']).fit(cov_type='HC1')
            print(f"  {cat_lbl} - {t_lbl}: {model.params['Intercept']:.2f} ± {model.bse['Intercept']:.2f}")
