import pickle
import os

subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0']).copy()

total_w = df['SPFWT0'].sum()

print(f"Total Analytical Sample: N={len(df)} (100.0%)")

print("\n--- LEVEL 1: TRACK ---")
for t_val, t_lbl in {0.0: "Academic", 1.0: "Vocational"}.items():
    sub = df[df['ED_GROUP'] == t_val]
    w = sub['SPFWT0'].sum()
    print(f"{t_lbl}: N={len(sub)} ({w/total_w*100:.2f}%)")

print("\n--- LEVEL 2: TRACK x GENDER ---")
for t_val, t_lbl in {0.0: "Academic", 1.0: "Vocational"}.items():
    for g_val, g_lbl in {1.0: "Male", 2.0: "Female"}.items():
        sub = df[(df['ED_GROUP'] == t_val) & (df['GENDER_R'] == g_val)]
        w = sub['SPFWT0'].sum()
        print(f"{t_lbl} -> {g_lbl}: N={len(sub)} ({w/total_w*100:.2f}%)")

print("\n--- LEVEL 3: TRACK x GENDER x SES ---")
for t_val, t_lbl in {0.0: "Academic", 1.0: "Vocational"}.items():
    for g_val, g_lbl in {1.0: "Male", 2.0: "Female"}.items():
        for s_val, s_lbl in {1.0: "Low SES", 2.0: "Med SES", 3.0: "High SES"}.items():
            sub = df[(df['ED_GROUP'] == t_val) & (df['GENDER_R'] == g_val) & (df['PAREDC2'] == s_val)]
            w = sub['SPFWT0'].sum()
            print(f"{t_lbl} -> {g_lbl} -> {s_lbl}: N={len(sub)} ({w/total_w*100:.2f}%)")

print("\n--- LEVEL 4: TRACK x GENDER x SES x MIGRATION ---")
for t_val, t_lbl in {0.0: "Academic", 1.0: "Vocational"}.items():
    for g_val, g_lbl in {1.0: "Male", 2.0: "Female"}.items():
        for s_val, s_lbl in {1.0: "Low SES", 2.0: "Med SES", 3.0: "High SES"}.items():
            for m_val, m_lbl in {1.0: "Native", 2.0: "Foreign"}.items():
                sub = df[(df['ED_GROUP'] == t_val) & (df['GENDER_R'] == g_val) & (df['PAREDC2'] == s_val) & (df['A2_Q03a_T'] == m_val)]
                w = sub['SPFWT0'].sum()
                print(f"{t_lbl} -> {g_lbl} -> {s_lbl} -> {m_lbl}: N={len(sub)} ({w/total_w*100:.2f}%)")
