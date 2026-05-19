import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
import pickle

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T', 'IMPARC2']
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
df = df.dropna(subset=anchors + workplace_vars + ['SPFWT0', 'AVG_LIT']).copy()

df['Female'] = (df['GENDER_R'] == 2.0).astype(int)
df['Foreign_Born'] = (df['A2_Q03a_T'] == 2.0).astype(int)

cols_for_corr = ['Female', 'Foreign_Born', 'PAREDC2', 'READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA', 'AVG_LIT']
labels_for_corr = ['Female', 'Foreign-Born', 'Parental SES', 'Work Reading', 'Work Writing', 'Literacy']

tracks = {0.0: 'Academic', 1.0: 'Vocational'}

for track_val, track_label in tracks.items():
    print(f"\n--- {track_label} ---")
    subset = df[df['ED_GROUP'] == track_val]
    
    d_stat = DescrStatsW(subset[cols_for_corr], weights=subset['SPFWT0'])
    corr_matrix = pd.DataFrame(d_stat.corrcoef, index=labels_for_corr, columns=labels_for_corr)
    
    # We mainly care about how Literacy correlates with Work Reading, Work Writing, and maybe SES/Demographics
    pairs = [
        ('Literacy', 'Work Reading'),
        ('Literacy', 'Work Writing'),
        ('Literacy', 'Parental SES'),
        ('Work Reading', 'Work Writing'),
        ('Female', 'Literacy'),
        ('Female', 'Work Reading'),
        ('Parental SES', 'Work Reading')
    ]
    
    for l1, l2 in pairs:
        i = labels_for_corr.index(l1)
        j = labels_for_corr.index(l2)
        c1 = cols_for_corr[i]
        c2 = cols_for_corr[j]
        
        r = corr_matrix.iloc[i, j]
        model = smf.wls(f"{c1} ~ {c2}", data=subset, weights=subset['SPFWT0']).fit(cov_type='HC1')
        p = model.pvalues[1]
        print(f"{l1} & {l2}: r = {r:.3f}, p = {p:.4f}")
