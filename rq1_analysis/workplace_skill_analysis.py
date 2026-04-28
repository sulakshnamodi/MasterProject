import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import zscore
import os
import pickle

# -----------------------------------------------------------------------------
# 1. SETUP AND DATA LOADING
# -----------------------------------------------------------------------------
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')
output_dir = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\rq1_analysis'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the pickle file containing the dataset
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe'].copy()

# -----------------------------------------------------------------------------
# 2. CLEANING, FILTERING & STANDARDIZATION
# -----------------------------------------------------------------------------
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T', 'IMPARC2']
workplace_vars = ['READWORKC2_WLE_CA_T1', 'WRITWORKC2_WLE_CA']
df = df.dropna(subset=anchors + ['SPFWT0'] + workplace_vars).copy()

# Standardize Workplace Variables (Z-scores) - matching user logic
df['READ_Z'] = zscore(df['READWORKC2_WLE_CA_T1'])
df['WRITE_Z'] = zscore(df['WRITWORKC2_WLE_CA'])

mapping_config = {
    'ED_GROUP': {0.0: 'Academic (ISCED 5A/6)', 1.0: 'Vocational (ISCED 5B)'}
}

COLOR_CONFIG = {
    'Academic': '#1b9e77',
    'Vocational': '#d95f02'
}

# -----------------------------------------------------------------------------
# SIGNIFICANCE HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_asterisks(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def draw_bracket(ax, x1, x2, y, h, text):
    """Helper function to draw statistical significance brackets."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color='black')
    weight = 'bold' if text != 'ns' else 'normal'
    fontsize = 12 if text != 'ns' else 10
    ax.text((x1+x2)/2, y+h+0.05, text, ha='center', va='bottom', color='black', 
            fontsize=fontsize, weight=weight)

# -----------------------------------------------------------------------------
# 3. ANALYSIS 3.1: MEAN SKILL USE BY TRACK & SIGNIFICANCE
# -----------------------------------------------------------------------------
print("--- ANALYSIS 3.1: WORKPLACE SKILL USE BY TRACK ---")
skills = {'READ_Z': 'Reading at Work', 'WRITE_Z': 'Writing at Work'}
results_3_1 = []
pval_dict = {}

for col, label in skills.items():
    print(f"\n=== {label} ===")
    pval_dict[col] = np.nan
    
    for track_val, track_label in mapping_config['ED_GROUP'].items():
        subset = df[df['ED_GROUP'] == track_val]
        model = smf.wls(f"{col} ~ 1", data=subset, weights=subset['SPFWT0']).fit(cov_type='HC1')
        
        results_3_1.append({
            'Skill': label,
            'Track': track_label,
            'Mean': model.params['Intercept'],
            'SEM': model.bse['Intercept']
        })
        print(f"  {track_label}: Mean Z = {model.params['Intercept']:.4f}, SEM = {model.bse['Intercept']:.4f}")
        
    # Track difference significance test
    model_sig = smf.wls(f"{col} ~ C(ED_GROUP)", data=df, weights=df['SPFWT0']).fit(cov_type='HC1')
    pval = model_sig.pvalues.get('C(ED_GROUP)[T.1.0]', np.nan)
    pval_dict[col] = pval
    print(f"  >> Track Difference Significance: P={pval:.4f}")

df_results = pd.DataFrame(results_3_1)

# -----------------------------------------------------------------------------
# 4. PLOTTING 3.2: GROUPED BAR PLOT FOR SKILL DEPLOYMENT (STYLED)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.35
categories = list(skills.values())
x_pos = np.arange(len(categories))

acad_data = df_results[df_results['Track'] == 'Academic (ISCED 5A/6)']
voc_data = df_results[df_results['Track'] == 'Vocational (ISCED 5B)']

acad_means = acad_data['Mean'].values
acad_sems = acad_data['SEM'].values

voc_means = voc_data['Mean'].values
voc_sems = voc_data['SEM'].values

ax.bar(x_pos - bar_width/2, acad_means, width=bar_width, yerr=acad_sems, capsize=5, 
       color=COLOR_CONFIG['Academic'], edgecolor='black', zorder=3, 
       error_kw={'elinewidth':1.2}, label='Academic (ISCED 5A/6)')
       
ax.bar(x_pos + bar_width/2, voc_means, width=bar_width, yerr=voc_sems, capsize=5, 
       color=COLOR_CONFIG['Vocational'], edgecolor='black', zorder=3, 
       error_kw={'elinewidth':1.2}, label='Vocational (ISCED 5B)')

# Draw significance brackets
for cat_idx, col in enumerate(skills.keys()):
    pval_cat = pval_dict[col]
    if not np.isnan(pval_cat):
        ast_cat = get_asterisks(pval_cat)
        bracket_y_cat = max(acad_means[cat_idx] + acad_sems[cat_idx], voc_means[cat_idx] + voc_sems[cat_idx]) + 0.05
        draw_bracket(ax, cat_idx - bar_width/2, cat_idx + bar_width/2, bracket_y_cat, 0.05, ast_cat)

ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_title('Workplace Skill Use Intensity by Education Track', fontsize=14, pad=15)

# Copied Styles
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.set_ylabel('Mean Skill Use Intensity (Z-Score)', fontsize=14)

# Dynamically stretch Y axis bounds slightly above the brackets
all_max_y = max([m + s for m, s in zip(acad_means, acad_sems)] + [m + s for m, s in zip(voc_means, voc_sems)])
ax.set_ylim(bottom=-0.5, top=all_max_y + 0.3)

plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()

plot_path = os.path.join(output_dir, 'rq3_workplace_skill_use.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Workplace Plot saved to: {plot_path}")

print("\nAll skill analysis pipelines evaluated successfully.")
