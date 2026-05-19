import os
import pandas as pd
import numpy as np
import pickle
from pymer4.models import Lmer

os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.3'
os.environ['R_LIBS_USER'] = r'C:\Users\brind\AppData\Local\R\win-library\4.5'
os.environ['PATH'] = r'C:\Program Files\R\R-4.5.3\bin\x64;' + os.environ['PATH']

subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

analysis_df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
analysis_df = analysis_df.dropna(subset=anchors + workplace_vars + ['SPFWT0']).copy()

analysis_df['intersectional_id'] = (
    analysis_df['ED_GROUP'].astype(str) + "_" +
    analysis_df['GENDER_R'].astype(str) + "_" +
    analysis_df['PAREDC2'].astype(str) + "_" +
    analysis_df['IMPARC2'].astype(str) + "_" +
    analysis_df['A2_Q03a_T'].astype(str)
)

valid_ids = analysis_df['intersectional_id'].value_counts()
valid_ids = valid_ids[valid_ids >= 5].index
analysis_df = analysis_df[analysis_df['intersectional_id'].isin(valid_ids)]

analysis_df['WGT_NORM'] = analysis_df['SPFWT0'] * (len(analysis_df) / analysis_df['SPFWT0'].sum())
from scipy.stats import zscore
analysis_df['READ_Z'] = zscore(analysis_df['READWORKC2_WLE_CA_T1'])
analysis_df['WRITE_Z'] = zscore(analysis_df['WRITWORKC2_WLE_CA'])

pv_columns = [f'PVLIT{i}' for i in range(1, 11)]
main_effects = "C(GENDER_R) + C(ED_GROUP) + C(PAREDC2) + C(IMPARC2) + C(A2_Q03a_T)"
rq3_effects = main_effects + " + READ_Z + WRITE_Z"

results_rq3_coefs = []

for pv in pv_columns:
    m_rq3 = Lmer(f"{pv} ~ {rq3_effects} + (1 | intersectional_id) + (0 + READ_Z | intersectional_id)", data=analysis_df)
    m_rq3.fit(weights="WGT_NORM", summarize=False)
    results_rq3_coefs.append(m_rq3.coefs)

# Rubin's Rules Pooling
pooled_rq3 = pd.concat(results_rq3_coefs).groupby(level=0).mean(numeric_only=True)
# We also want the standard error pooled
# For pooling SE, let's do a proper pooling
# Let's pool Estimate and Std.Error using Rubin's Rules
estimates_df = pd.concat([df['Estimate'] for df in results_rq3_coefs], axis=1)
se_df = pd.concat([df['SE'] for df in results_rq3_coefs], axis=1)

pooled_estimates = estimates_df.mean(axis=1)
W_var = (se_df ** 2).mean(axis=1)
B_var = estimates_df.var(axis=1)
M = len(pv_columns)
total_var = W_var + (1 + 1/M) * B_var
pooled_se = np.sqrt(total_var)

# Z-score and P-values
pooled_z = pooled_estimates / pooled_se
from scipy.stats import norm
pooled_p = 2 * (1 - norm.cdf(np.abs(pooled_z)))

final_df = pd.DataFrame({
    'Estimate': pooled_estimates,
    'SE': pooled_se,
    'Z': pooled_z,
    'P': pooled_p
})

print(final_df.to_string())
