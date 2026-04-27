import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.formula.api as smf
import os
import pickle

# -----------------------------------------------------------------------------
# 1. SETUP AND DATA LOADING
# -----------------------------------------------------------------------------
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')
output_dir = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\rq1_analysis'

os.makedirs(output_dir, exist_ok=True)

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()

# Average Plausible Values
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

# -----------------------------------------------------------------------------
# 2. CLEANING & PREPROCESSING
# -----------------------------------------------------------------------------
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T']
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
df = df.dropna(subset=anchors + workplace_vars + ['SPFWT0', 'AVG_LIT']).copy()

df['Female'] = (df['GENDER_R'] == 2.0).astype(int)
df['Foreign_Born'] = (df['A2_Q03a_T'] == 2.0).astype(int)

cols_for_corr = ['Female', 'Foreign_Born', 'PAREDC2', 'READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA', 'AVG_LIT']
labels_for_corr = ['Female', 'Foreign-Born', 'Parental SES', 'Work Reading', 'Work Writing', 'Literacy']

# Helper for significance stars
def get_stars(r, p):
    if pd.isna(p): return ""
    if p < 0.001: return f"{r:.2f}***"
    if p < 0.01: return f"{r:.2f}**"
    if p < 0.05: return f"{r:.2f}*"
    return f"{r:.2f}"

# -----------------------------------------------------------------------------
# 3. ANALYSIS 4.1: COMPUTE WEIGHTED CORRELATIONS & P-VALUES BY TRACK
# -----------------------------------------------------------------------------
print("--- ANALYSIS 4.1: WEIGHTED CORRELATION HEATMAPS BY TRACK ---")

tracks = {0.0: 'Academic (ISCED 5A/6)', 1.0: 'Vocational (ISCED 5B)'}

for track_val, track_label in tracks.items():
    print(f"\nProcessing {track_label}...")
    subset = df[df['ED_GROUP'] == track_val]
    
    # 1. Compute Weighted Correlation Matrix
    d_stat = DescrStatsW(subset[cols_for_corr], weights=subset['SPFWT0'])
    corr_matrix = pd.DataFrame(d_stat.corrcoef, index=labels_for_corr, columns=labels_for_corr)
    
    # 2. Compute Weighted P-Values
    p_values = pd.DataFrame(np.zeros((6, 6)), index=labels_for_corr, columns=labels_for_corr)
    for i, col1 in enumerate(cols_for_corr):
        for j, col2 in enumerate(cols_for_corr):
            if i == j:
                p_values.iloc[i, j] = np.nan
            else:
                formula = f"{col1} ~ {col2}"
                model = smf.wls(formula, data=subset, weights=subset['SPFWT0']).fit(cov_type='HC1')
                p_values.iloc[i, j] = model.pvalues[1]
                
    # 3. Create Annotations
    annot_matrix = pd.DataFrame("", index=labels_for_corr, columns=labels_for_corr)
    for i in range(6):
        for j in range(6):
            if i != j:
                r = corr_matrix.iloc[i, j]
                p = p_values.iloc[i, j]
                annot_matrix.iloc[i, j] = get_stars(r, p)
                
    # -------------------------------------------------------------------------
    # 4. PLOTTING HEATMAP
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    mask = np.eye(6, dtype=bool)
    
    # Calculate bounds
    off_diag = corr_matrix.values[~mask]
    max_corr = np.abs(off_diag).max()
    v_bound = float(np.ceil(max_corr * 10) / 10)
    
    ax = sns.heatmap(
        corr_matrix, 
        mask=mask, 
        annot=annot_matrix, 
        fmt="", 
        cmap='BrBG', 
        vmin=-v_bound, 
        vmax=v_bound, 
        center=0,
        linewidths=1, 
        linecolor='black',
        annot_kws={"size": 12}
    )
    
    # Styling matching descriptive_analysis/correlation_heatmap.py
    ax.set_title(f'Correlation Heatmap: {track_label}', fontsize=14, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, weight='normal', rotation=20, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, weight='normal', rotation=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Pearson Correlation ($r$)', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    fig = plt.gcf()
    fig.text(0.05, 0.03, "* p < 0.05, ** p < 0.01, *** p < 0.001 (Survey-weighted p-values).", 
             fontsize=10, style='italic')
             
    track_str = "academic" if track_val == 0.0 else "vocational"
    plot_path = os.path.join(output_dir, f'rq4_1_correlation_heatmap_{track_str}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

print("\nAll separated correlation maps updated.")
