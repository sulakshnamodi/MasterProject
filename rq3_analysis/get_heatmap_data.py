import pickle
import pandas as pd
from scipy.stats import zscore

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
df = df.dropna(subset=anchors + workplace_vars + ['SPFWT0']).copy()

df['intersectional_id'] = (
    df['ED_GROUP'].astype(str) + "_" +
    df['GENDER_R'].astype(str) + "_" +
    df['PAREDC2'].astype(str) + "_" +
    df['IMPARC2'].astype(str) + "_" +
    df['A2_Q03a_T'].astype(str)
)

valid_ids = df['intersectional_id'].value_counts()
valid_ids = valid_ids[valid_ids >= 5].index
df = df[df['intersectional_id'].isin(valid_ids)].copy()

df['READ_Z'] = zscore(df['READWORKC2_WLE_CA_T1'])
df['WRITE_Z'] = zscore(df['WRITWORKC2_WLE_CA'])

def label_strata(strat_id):
    parts = strat_id.split('_')
    edu = "Univ" if parts[0] == '0.0' else "Voc"
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses_map = {'1': 'Low', '2': 'Med', '3': 'High'}
    ses = ses_map.get(parts[2][0], 'Low')
    mig = "Nat" if parts[4] == '1.0' else "For"
    return f"{edu}-{gen}-{ses}-{mig}"

access_df = df.groupby('intersectional_id').agg({
    'READ_Z': 'mean',
    'WRITE_Z': 'mean',
}).reset_index()
access_df['label'] = access_df['intersectional_id'].apply(label_strata)

print("Top 5 READ_Z:")
print(access_df.sort_values('READ_Z', ascending=False)[['label', 'READ_Z', 'WRITE_Z']].head(5).to_string(index=False))
print("\nBottom 5 READ_Z:")
print(access_df.sort_values('READ_Z', ascending=False)[['label', 'READ_Z', 'WRITE_Z']].tail(5).to_string(index=False))
