import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pickle

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0', 'AVG_LIT']).copy()

# Ensure ACADEMIC is the predictor for the advantage
df['ACADEMIC'] = np.where(df['ED_GROUP'] == 0.0, 1, 0)

mapping_config = {
    'GENDER_R': {1.0: 'Male', 2.0: 'Female'},
    'A2_Q03a_T': {1.0: 'Born in Norway', 2.0: 'Foreign-born'},
    'PAREDC2': {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'},
    'IMPARC2': {1.0: 'Both Foreign', 2.0: 'One Foreign', 3.0: 'Both Native'}
}

demographics = ['GENDER_R', 'A2_Q03a_T', 'PAREDC2', 'IMPARC2']
for col in demographics:
    print(f"\n--- {col} ---")
    for cat_val, cat_label in mapping_config[col].items():
        subset_sub = df[df[col] == cat_val].copy()
        
        # Means and SE
        acad_sub = subset_sub[subset_sub['ED_GROUP'] == 0.0]
        voc_sub = subset_sub[subset_sub['ED_GROUP'] == 1.0]
        
        if len(acad_sub) > 0 and len(voc_sub) > 0:
            m_acad = smf.wls("AVG_LIT ~ 1", data=acad_sub, weights=acad_sub['SPFWT0']).fit(cov_type='HC1')
            m_voc = smf.wls("AVG_LIT ~ 1", data=voc_sub, weights=voc_sub['SPFWT0']).fit(cov_type='HC1')
            
            # Regression for difference
            model_diff = smf.wls("AVG_LIT ~ ACADEMIC", data=subset_sub, weights=subset_sub['SPFWT0']).fit(cov_type='HC1')
            beta = model_diff.params['ACADEMIC']
            se = model_diff.bse['ACADEMIC']
            conf_int = model_diff.conf_int().loc['ACADEMIC']
            pval = model_diff.pvalues['ACADEMIC']
            
            print(f"{cat_label}: Acad(M={m_acad.params['Intercept']:.2f}, SE={m_acad.bse['Intercept']:.2f}) | Voc(M={m_voc.params['Intercept']:.2f}, SE={m_voc.bse['Intercept']:.2f})")
            print(f"   Beta={beta:.2f}, SE={se:.2f}, 95% CI [{conf_int[0]:.2f}, {conf_int[1]:.2f}], p={pval:.4f}")
        else:
            print(f"{cat_label}: NOT ENOUGH DATA IN BOTH TRACKS")
