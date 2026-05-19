import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)]
analysis_df['PV_AVG'] = analysis_df[pv_columns].mean(axis=1)

def label_strata(row):
    parts = row['intersectional_id'].split('_')
    gen = "Male" if parts[1] == '1.0' else "Fem"
    ses_map = {'1': 'Low', '2': 'Med', '3': 'High'}
    ses = ses_map.get(parts[2][0], 'Low')
    mig = "Nat" if parts[4] == '1.0' else "For"
    return f"{gen}_{ses}_{mig}"

analysis_df['label'] = analysis_df.apply(label_strata, axis=1)
order_labels = analysis_df.groupby('label')['PV_AVG'].mean().sort_values(ascending=False).index

plt.figure(figsize=(11, 13))
colors = {0.0: "#1b9e77", 1.0: "#d95f02"}
# Try with dodge=True
sns.barplot(x='PV_AVG', y='label', data=analysis_df, hue='ED_GROUP', dodge=True, palette=colors, order=order_labels, errorbar=('ci', 95))
plt.axvline(analysis_df['PV_AVG'].mean(), color='black', linestyle='--')
plt.title('Test Dodge=True')
plt.tight_layout()
plt.savefig('test_dodge_true.png')
plt.close()

# Try with dodge=False
plt.figure(figsize=(11, 13))
sns.barplot(x='PV_AVG', y='label', data=analysis_df, hue='ED_GROUP', dodge=False, palette=colors, order=order_labels, errorbar=('ci', 95))
plt.axvline(analysis_df['PV_AVG'].mean(), color='black', linestyle='--')
plt.title('Test Dodge=False')
plt.tight_layout()
plt.savefig('test_dodge_false.png')
plt.close()

print("Test plots generated.")
