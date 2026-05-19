import pandas as pd
import numpy as np
import pickle 

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

analysis_df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
pv_columns = [f'PVLIT{i}' for i in range(1, 11)]

analysis_df = analysis_df.dropna(subset=anchors + pv_columns + ['SPFWT0']).copy()
analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

counts = analysis_df['intersectional_id'].value_counts()
valid_ids = counts[counts >= 5].index
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)].copy()

analysis_df['PV_AVG'] = analysis_df[pv_columns].mean(axis=1)

def label_strata(row):
    parts = row['intersectional_id'].split('_')
    edu = "Univ" if parts[0] == '0.0' else "Voc"
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses_map = {'1': 'Low', '2': 'Med', '3': 'High'}
    ses = ses_map.get(parts[2][0], 'Low')
    mig = "Native" if parts[4] == '1.0' else "Abroad"
    return f"{edu}-{gen}-{ses}-{mig}"

analysis_df['label'] = analysis_df.apply(label_strata, axis=1)

from scipy import stats
def sem_weighted(x):
    return stats.sem(x, ddof=1)

res = analysis_df.groupby('label')['PV_AVG'].agg(['mean', sem_weighted, 'count']).sort_values('mean', ascending=False)
res['ci_margin'] = res['sem_weighted'] * 1.96
print(res.head(5).to_string())
print(res.tail(5).to_string())

grand_mean = analysis_df['PV_AVG'].mean()
print(f"Grand Mean: {grand_mean:.2f}")
