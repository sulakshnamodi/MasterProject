import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
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

# Average the 10 Plausible Values (PVs) for literacy
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

# -----------------------------------------------------------------------------
# 2. CLEANING & FILTERING
# -----------------------------------------------------------------------------
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0', 'AVG_LIT']).copy()

mapping_config = {
    'ED_GROUP': {0.0: 'Academic (ISCED 5A/6)', 1.0: 'Vocational (ISCED 5B)'},
    'GENDER_R': {1.0: 'Male', 2.0: 'Female'},
    'A2_Q03a_T': {1.0: 'Born in Norway', 2.0: 'Foreign-born'},
    'PAREDC2': {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'},
    'IMPARC2': {1.0: 'Both Foreign', 2.0: 'One Foreign', 3.0: 'Both Native'}
}

demographics = ['GENDER_R', 'A2_Q03a_T', 'PAREDC2', 'IMPARC2']
title_map = {
    'GENDER_R': 'Gender',
    'A2_Q03a_T': 'Migration',
    'PAREDC2': 'SES',
    'IMPARC2': 'Parents Migration'
}

COLOR_CONFIG = {
    'Academic': '#1b9e77',
    'Vocational': '#d95f02'
}

Y_MIN = 230
Y_MAX = 350

# -----------------------------------------------------------------------------
# SIGNIFICANCE HELPER FUNCTIONS (from mean_literacy_by_strata.py)
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
    ax.text((x1+x2)/2, y+h+1.5, text, ha='center', va='bottom', color='black', 
            fontsize=fontsize, weight=weight)

# -----------------------------------------------------------------------------
# 3. ANALYSIS 2.1: OVERALL MEAN PROFICIENCY BY TRACK & SIGNIFICANCE
# -----------------------------------------------------------------------------
print("--- ANALYSIS 2.1: OVERALL MEAN LITERACY BY TRACK ---")
overall_results = []

for track_val, track_label in mapping_config['ED_GROUP'].items():
    subset = df[df['ED_GROUP'] == track_val]
    model = smf.wls("AVG_LIT ~ 1", data=subset, weights=subset['SPFWT0']).fit(cov_type='HC1')
    
    overall_results.append({
        'Track': track_label,
        'Mean': model.params['Intercept'],
        'SEM': model.bse['Intercept']
    })
    print(f"  {track_label}: Mean = {model.params['Intercept']:.2f}, SEM = {model.bse['Intercept']:.2f}")

# Test significance
model_sig = smf.wls("AVG_LIT ~ C(ED_GROUP)", data=df, weights=df['SPFWT0']).fit(cov_type='HC1')
pval_overall = model_sig.pvalues.get('C(ED_GROUP)[T.1.0]', np.nan)

# -----------------------------------------------------------------------------
# 4. ANALYSIS 2.2: STRATIFIED PROFICIENCY BY DEMOGRAPHICS
# -----------------------------------------------------------------------------
print("\n--- ANALYSIS 2.2: STRATIFIED PROFICIENCY GAPS ---")
strat_results = []
pval_dict = {}

for col in demographics:
    print(f"\n=== {title_map[col]} ===")
    col_mapping = mapping_config[col]
    pval_dict[col] = {}
    
    for cat_val, cat_label in col_mapping.items():
        for track_val, track_label in mapping_config['ED_GROUP'].items():
            subset = df[(df[col] == cat_val) & (df['ED_GROUP'] == track_val)]
            
            if len(subset) > 0:
                model = smf.wls("AVG_LIT ~ 1", data=subset, weights=subset['SPFWT0']).fit(cov_type='HC1')
                mean_val = model.params['Intercept']
                sem_val = model.bse['Intercept']
            else:
                mean_val = np.nan
                sem_val = np.nan
                
            strat_results.append({
                'Demographic': col,
                'Category': cat_label,
                'Track': track_label,
                'Mean': mean_val,
                'SEM': sem_val
            })
            
        # Significance test within this subgroup
        subset_sub = df[df[col] == cat_val]
        if len(subset_sub['ED_GROUP'].unique()) > 1:
            model_sub = smf.wls("AVG_LIT ~ C(ED_GROUP)", data=subset_sub, weights=subset_sub['SPFWT0']).fit(cov_type='HC1')
            pval_sub = model_sub.pvalues.get('C(ED_GROUP)[T.1.0]', np.nan)
            pval_dict[col][cat_label] = pval_sub
            print(f"  {cat_label} Track Gap P-Value: P={pval_sub:.4f}")
        else:
            pval_dict[col][cat_label] = np.nan

df_strat = pd.DataFrame(strat_results)

# -----------------------------------------------------------------------------
# 5. PLOTTING 2.1: OVERALL MEAN LITERACY BY TRACK
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

tracks = [r['Track'] for r in overall_results]
means = [r['Mean'] for r in overall_results]
sems = [r['SEM'] for r in overall_results]
x_pos = np.arange(len(tracks))

ax.bar(x_pos, means, yerr=sems, width=0.6, capsize=5, 
       color=[COLOR_CONFIG['Academic'], COLOR_CONFIG['Vocational']], 
       edgecolor='black', zorder=3, error_kw={'elinewidth':1.2})

ax.set_xticks(x_pos)
ax.set_xticklabels(tracks, fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_title('Overall Mean Literacy by Education Track', fontsize=14, pad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_ylabel('Mean Literacy Score', fontsize=14)

# Draw overall significance bracket
ast_overall = get_asterisks(pval_overall)
bracket_y_overall = max([m + s for m, s in zip(means, sems)]) + 5
draw_bracket(ax, 0, 1, bracket_y_overall, 5, ast_overall)

plt.tight_layout()
plot1_path = os.path.join(output_dir, 'rq2_1_overall_literacy_by_track.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"\nOverall Plot saved to: {plot1_path}")

# -----------------------------------------------------------------------------
# 6. PLOTTING 2.2: STRATIFIED LITERACY BY DEMOGRAPHICS (2x2)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
axes = axes.flatten()
bar_width = 0.35

for idx, col in enumerate(demographics):
    ax = axes[idx]
    col_mapping = mapping_config[col]
    categories = list(col_mapping.values())
    x_pos = np.arange(len(categories))
    
    acad_data = df_strat[(df_strat['Demographic'] == col) & (df_strat['Track'] == 'Academic (ISCED 5A/6)')]
    voc_data = df_strat[(df_strat['Demographic'] == col) & (df_strat['Track'] == 'Vocational (ISCED 5B)')]
    
    acad_means = [acad_data[acad_data['Category'] == cat]['Mean'].values[0] if len(acad_data[acad_data['Category'] == cat]) > 0 else 0 for cat in categories]
    acad_sems = [acad_data[acad_data['Category'] == cat]['SEM'].values[0] if len(acad_data[acad_data['Category'] == cat]) > 0 else 0 for cat in categories]
    
    voc_means = [voc_data[voc_data['Category'] == cat]['Mean'].values[0] if len(voc_data[voc_data['Category'] == cat]) > 0 else 0 for cat in categories]
    voc_sems = [voc_data[voc_data['Category'] == cat]['SEM'].values[0] if len(voc_data[voc_data['Category'] == cat]) > 0 else 0 for cat in categories]
    
    ax.bar(x_pos - bar_width/2, acad_means, width=bar_width, yerr=acad_sems, capsize=5, 
           color=COLOR_CONFIG['Academic'], edgecolor='black', zorder=3, 
           error_kw={'elinewidth':1.2}, label='Academic')
           
    ax.bar(x_pos + bar_width/2, voc_means, width=bar_width, yerr=voc_sems, capsize=5, 
           color=COLOR_CONFIG['Vocational'], edgecolor='black', zorder=3, 
           error_kw={'elinewidth':1.2}, label='Vocational')
           
    # Draw significance brackets for each category
    for cat_idx, cat in enumerate(categories):
        pval_cat = pval_dict[col].get(cat, np.nan)
        if not np.isnan(pval_cat):
            ast_cat = get_asterisks(pval_cat)
            bracket_y_cat = max(acad_means[cat_idx] + acad_sems[cat_idx], voc_means[cat_idx] + voc_sems[cat_idx]) + 5
            draw_bracket(ax, cat_idx - bar_width/2, cat_idx + bar_width/2, bracket_y_cat, 3, ast_cat)
            
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title(title_map[col], fontsize=14, pad=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.set_ylim(Y_MIN, Y_MAX)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.02))
fig.supylabel('Mean Literacy Score', fontsize=14, x=0.01)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12, left=0.08, wspace=0.2, hspace=0.3)

plot2_path = os.path.join(output_dir, 'rq2_2_stratified_literacy_by_track.png')
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"Stratified Plot saved to: {plot2_path}")

print("\nAll proficiency analyses completed successfully!")
