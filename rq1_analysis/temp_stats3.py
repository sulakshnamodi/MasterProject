import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import zscore
import pickle

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T', 'IMPARC2']
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
df = df.dropna(subset=anchors + ['SPFWT0'] + workplace_vars).copy()

df['READ_Z'] = zscore(df['READWORKC2_WLE_CA_T1'])
df['WRITE_Z'] = zscore(df['WRITWORKC2_WLE_CA'])
df['ACADEMIC'] = np.where(df['ED_GROUP'] == 0.0, 1, 0)

skills = {'READ_Z': 'Reading at Work', 'WRITE_Z': 'Writing at Work'}

for col, label in skills.items():
    print(f"\n=== {label} ===")
    acad_sub = df[df['ED_GROUP'] == 0.0]
    voc_sub = df[df['ED_GROUP'] == 1.0]
    
    m_acad = smf.wls(f"{col} ~ 1", data=acad_sub, weights=acad_sub['SPFWT0']).fit(cov_type='HC1')
    m_voc = smf.wls(f"{col} ~ 1", data=voc_sub, weights=voc_sub['SPFWT0']).fit(cov_type='HC1')
    
    model_diff = smf.wls(f"{col} ~ ACADEMIC", data=df, weights=df['SPFWT0']).fit(cov_type='HC1')
    beta = model_diff.params['ACADEMIC']
    se = model_diff.bse['ACADEMIC']
    conf_int = model_diff.conf_int().loc['ACADEMIC']
    pval = model_diff.pvalues['ACADEMIC']
    
    print(f"Acad(M={m_acad.params['Intercept']:.2f}, SE={m_acad.bse['Intercept']:.2f}) | Voc(M={m_voc.params['Intercept']:.2f}, SE={m_voc.bse['Intercept']:.2f})")
    print(f"Beta={beta:.2f}, SE={se:.2f}, 95% CI [{conf_int[0]:.2f}, {conf_int[1]:.2f}], p={pval:.4f}")
