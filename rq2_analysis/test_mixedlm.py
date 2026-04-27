"""
MixedLM Multilevel Diagnostics Script

Validates multilevel modeling parameters and checks 
computational functionality under local library allocations.
"""

import pandas as pd
import numpy as np
import pickle
import os
import statsmodels.formula.api as smf

# Data extraction setup
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()

# Establish literacy skills proxy
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

# Filter anchors
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['AVG_LIT']).copy()

# Create composite nested criteria
df['intersectional_id'] = (
    df['ED_GROUP'].astype(str) + "_" +
    df['GENDER_R'].astype(str) + "_" +
    df['PAREDC2'].astype(str) + "_" +
    df['IMPARC2'].astype(str) + "_" +
    df['A2_Q03a_T'].astype(str)
)

# Subset large configurations
counts = df['intersectional_id'].value_counts()
valid_ids = counts[counts >= 5].index
df = df[df['intersectional_id'].isin(valid_ids)]

print(f"Evaluating parameters smoothly...")

# Execute standard unweighted baseline mapping
model = smf.mixedlm("AVG_LIT ~ 1", data=df, groups=df["intersectional_id"]).fit()
var_between = float(model.cov_re.iloc[0, 0])
var_resid = float(model.scale)
vpc = var_between / (var_between + var_resid)

print(f"VPC Baseline Metric: {vpc:.4f}")
