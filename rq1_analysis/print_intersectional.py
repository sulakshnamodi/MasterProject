import pickle
import os

subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T', 'IMPARC2']
df = df.dropna(subset=anchors + ['SPFWT0']).copy()

def create_intersectional_label(row):
    gen = 'M' if row['GENDER_R'] == 1.0 else 'F'
    mig = 'Nat' if row['A2_Q03a_T'] == 1.0 else 'For'
    ses = 'Low' if row['PAREDC2'] == 1.0 else ('Med' if row['PAREDC2'] == 2.0 else 'High')
    return f"{gen}-{mig}-{ses}"

df['intersectional_strata'] = df.apply(create_intersectional_label, axis=1)

total_w = df['SPFWT0'].sum()

for track_val, track_lbl in {0.0: "Academic", 1.0: "Vocational"}.items():
    sub_t = df[df['ED_GROUP'] == track_val]
    t_w = sub_t['SPFWT0'].sum()
    print(f"\n--- Top Intersectional Strata in {track_lbl} ---")
    
    counts = sub_t.groupby('intersectional_strata')['SPFWT0'].sum().reset_index()
    counts['pct'] = counts['SPFWT0'] / t_w * 100
    counts = counts.sort_values(by='pct', ascending=False).head(5)
    for idx, row in counts.iterrows():
        print(f"  {row['intersectional_strata']}: {row['pct']:.2f}%")
