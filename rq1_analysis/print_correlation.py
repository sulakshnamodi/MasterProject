import pickle
import os
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T']
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
df = df.dropna(subset=anchors + workplace_vars + ['SPFWT0', 'AVG_LIT']).copy()

df['Female'] = (df['GENDER_R'] == 2.0).astype(int)
df['Foreign_Born'] = (df['A2_Q03a_T'] == 2.0).astype(int)

cols_for_corr = ['Female', 'Foreign_Born', 'PAREDC2', 'READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA', 'AVG_LIT']
labels_for_corr = ['Female', 'Foreign-Born', 'Parental SES', 'Work Reading', 'Work Writing', 'Literacy']

for track_val, track_lbl in {0.0: "Academic", 1.0: "Vocational"}.items():
    sub = df[df['ED_GROUP'] == track_val]
    d_stat = DescrStatsW(sub[cols_for_corr], weights=sub['SPFWT0'])
    corr = pd.DataFrame(d_stat.corrcoef, index=labels_for_corr, columns=labels_for_corr)
    
    print(f"\n=== {track_lbl} ===")
    print(f"  Literacy vs Work Reading: r = {corr.loc['Literacy', 'Work Reading']:.2f}")
    print(f"  Literacy vs Work Writing: r = {corr.loc['Literacy', 'Work Writing']:.2f}")
    print(f"  Work Reading vs Work Writing: r = {corr.loc['Work Reading', 'Work Writing']:.2f}")
