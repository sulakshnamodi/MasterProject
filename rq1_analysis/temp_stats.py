import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle
import os

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()

# Average the 10 Plausible Values (PVs) for literacy
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0', 'AVG_LIT']).copy()

# Means and SEs
for track_val, track_label in {0.0: 'Academic', 1.0: 'Vocational'}.items():
    subset = df[df['ED_GROUP'] == track_val]
    model = smf.wls("AVG_LIT ~ 1", data=subset, weights=subset['SPFWT0']).fit(cov_type='HC1')
    print(f"{track_label}: M = {model.params['Intercept']:.2f}, SE = {model.bse['Intercept']:.2f}")

# Regression to find the difference
# We want academic enrollment as the predictor.
# Currently ED_GROUP is 0.0=Academic, 1.0=Vocational. 
# Let's make Academic=1, Vocational=0 so beta represents the 'academic advantage' as in the example.
df['ACADEMIC'] = np.where(df['ED_GROUP'] == 0.0, 1, 0)
model_sig = smf.wls("AVG_LIT ~ ACADEMIC", data=df, weights=df['SPFWT0']).fit(cov_type='HC1')
print(model_sig.summary())
