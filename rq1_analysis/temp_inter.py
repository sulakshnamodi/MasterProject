import pandas as pd
import pickle
import os

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
loaded_data = pickle.load(open(subdataset_filepath, 'rb'))
df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0']).copy()

def create_intersectional_label(row):
    gen = 'M' if row['GENDER_R'] == 1.0 else 'F'
    mig = 'Nat' if row['A2_Q03a_T'] == 1.0 else 'For'
    ses = 'Low' if row['PAREDC2'] == 1.0 else ('Med' if row['PAREDC2'] == 2.0 else 'High')
    return f"{gen}-{ses}-{mig}"

df['intersectional_strata'] = df.apply(create_intersectional_label, axis=1)

mapping_config = {'ED_GROUP': {0.0: 'Academic', 1.0: 'Vocational'}}
results_1_2 = []

for track_val, track_label in mapping_config['ED_GROUP'].items():
    df_track = df[df['ED_GROUP'] == track_val]
    track_weight_sum = df_track['SPFWT0'].sum()
    strata_counts = df_track.groupby('intersectional_strata')['SPFWT0'].sum().reset_index()
    strata_counts['Weighted %'] = (strata_counts['SPFWT0'] / track_weight_sum) * 100
    strata_counts['Track'] = track_label
    results_1_2.append(strata_counts)

df_results_1_2 = pd.concat(results_1_2)
top_strata = df_results_1_2.groupby('intersectional_strata')['Weighted %'].mean().sort_values(ascending=False).head(10).index

plot_data = df_results_1_2[df_results_1_2['intersectional_strata'].isin(top_strata)]
print(plot_data.sort_values(by=['Weighted %'], ascending=False).to_string())
