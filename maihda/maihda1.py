""" RQ3: To what extent does intersectional group membership, based on education type, gender, socioeconomic status and migration background, account for differences in workplace
cognitive literacy skill application among Norwegian adults?  """

import os, sys
import pandas as pd
import numpy as np
import pickle 
from scipy.stats import norm  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf


# RQ1: How do the levels of cognitive skills (literacy, numeracy, problem-solving) compare between university and vocationally educated adults at work in Norway?

# Setting of subdataset file
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\regression'

#load the pickle file back
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)


subdataset_df = loaded_data['dataframe']

# 1. Drop rows with missing values in your intersectional anchors
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
analysis_df = subdataset_df.dropna(subset=anchors).copy()

for col in analysis_df.columns:
    if col not in anchors:
        continue
    unique_vals = analysis_df[col].unique()
    print(f"Column: {col} | Unique Values: {unique_vals}")


metadata = loaded_data['metadata'] # dictionary from the codebook

# 2. Create the Intersectional Strata ID
analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

print(f"Number of unique intersectional strata: {analysis_df['intersectional_id'].nunique()}")

# This counts how many people are in every unique combination
counts = analysis_df.groupby(['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T' ]).size()
print(counts)
print(f"Total number of non-empty strata: {len(counts)}")

# Calculate number of empty strata
total_theoretical_strata = 1
for col in anchors:
    total_theoretical_strata *= analysis_df[col].nunique()

print(f"Total theoretical strata: {total_theoretical_strata}")
print(f"Number of empty strata: {total_theoretical_strata - len(counts)}")





