import pandas as pd
import pickle
import os
subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
loaded_data = pickle.load(open(subdataset_filepath, 'rb'))
df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0'])
mapping_config = {'ED_GROUP': {0.0: 'Academic', 1.0: 'Vocational'}, 'GENDER_R': {1.0: 'Male', 2.0: 'Female'}, 'A2_Q03a_T': {1.0: 'Born in Norway', 2.0: 'Foreign-born'}, 'PAREDC2': {1.0: 'Low SES', 2.0: 'Med SES', 3.0: 'High SES'}, 'IMPARC2': {1.0: 'Both Foreign', 2.0: 'One Foreign', 3.0: 'Both Native'}}
demographics = ['GENDER_R', 'PAREDC2', 'A2_Q03a_T']
for col in demographics:
    print(f"\n--- {col} ---")
    for track_val, track_label in mapping_config['ED_GROUP'].items():
        df_t = df[df['ED_GROUP'] == track_val]
        w_sum = df_t['SPFWT0'].sum()
        for cat_val, cat_label in mapping_config[col].items():
            sub = df_t[df_t[col] == cat_val]
            pct = (sub['SPFWT0'].sum() / w_sum) * 100
            print(f'{track_label} - {cat_label}: {pct:.1f}% (N={len(sub)})')
