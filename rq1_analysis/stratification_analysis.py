import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# -----------------------------------------------------------------------------
# 2. CLEANING & FILTERING
# -----------------------------------------------------------------------------
# Ensure all analytical anchors and weights are non-null
anchors = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'IMPARC2', 'A2_Q03a_T']
df = df.dropna(subset=anchors + ['SPFWT0']).copy()

# Value Mappings for readable output and plots
mapping_config = {
    'ED_GROUP': {0.0: 'Academic (ISCED 5A/6)', 1.0: 'Vocational (ISCED 5B)'},
    'GENDER_R': {1.0: 'Male', 2.0: 'Female'},
    'A2_Q03a_T': {1.0: 'Born in Norway', 2.0: 'Foreign-born'},
    'PAREDC2': {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'},
    'IMPARC2': {1.0: 'Both Foreign', 2.0: 'One Foreign', 3.0: 'Both Native'}
}

# -----------------------------------------------------------------------------
# 3. ANALYSIS 1.1: BIVARIATE DISTRIBUTIONS BY TRACK & SIGNIFICANCE
# -----------------------------------------------------------------------------
demographics = ['GENDER_R', 'A2_Q03a_T', 'PAREDC2', 'IMPARC2']
results_1_1 = []

for col in demographics:
    col_mapping = mapping_config[col]
    
    for track_val, track_label in mapping_config['ED_GROUP'].items():
        df_track = df[df['ED_GROUP'] == track_val]
        track_weight_sum = df_track['SPFWT0'].sum()
        
        for cat_val, cat_label in col_mapping.items():
            subset = df_track[df_track[col] == cat_val]
            weighted_pct = (subset['SPFWT0'].sum() / track_weight_sum) * 100
            unweighted_n = len(subset)
            
            results_1_1.append({
                'Demographic': col,
                'Category': cat_label,
                'Track': track_label,
                'Weighted %': weighted_pct
            })

df_results_1_1 = pd.DataFrame(results_1_1)

# -----------------------------------------------------------------------------
# 4. ANALYSIS 1.2: INTERSECTIONAL COMPOSITION BY TRACK
# -----------------------------------------------------------------------------
def create_intersectional_label(row):
    gen = 'M' if row['GENDER_R'] == 1.0 else 'F'
    mig = 'Nat' if row['A2_Q03a_T'] == 1.0 else 'For'
    ses = 'Low' if row['PAREDC2'] == 1.0 else ('Med' if row['PAREDC2'] == 2.0 else 'High')
    return f"{gen}-{mig}-{ses}"

df['intersectional_strata'] = df.apply(create_intersectional_label, axis=1)

results_1_2 = []
for track_val, track_label in mapping_config['ED_GROUP'].items():
    df_track = df[df['ED_GROUP'] == track_val]
    track_weight_sum = df_track['SPFWT0'].sum()
    
    strata_counts = df_track.groupby('intersectional_strata')['SPFWT0'].sum().reset_index()
    strata_counts['Weighted %'] = (strata_counts['SPFWT0'] / track_weight_sum) * 100
    strata_counts['Track'] = track_label
    results_1_2.append(strata_counts)

df_results_1_2 = pd.concat(results_1_2)

# -----------------------------------------------------------------------------
# 5. PLOTTING 1.1: BIVARIATE DEMOGRAPHICS BY TRACK (STYLED)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
axes = axes.flatten()

# Colors requested by User
colors = {'Academic (ISCED 5A/6)': '#1b9e77', 'Vocational (ISCED 5B)': '#d95f02'}

title_map = {
    'GENDER_R': 'Gender',
    'A2_Q03a_T': 'Migration',
    'PAREDC2':  'SES',
    'IMPARC2': 'Parents Migration'
}

for i, col in enumerate(demographics):
    ax = axes[i]
    plot_data = df_results_1_1[df_results_1_1['Demographic'] == col]
    
    sns.barplot(
        data=plot_data,
        x='Category',
        y='Weighted %',
        hue='Track',
        palette=colors,
        ax=ax,
        edgecolor='black',
        linewidth=1,
        zorder=3
    )
    
    # Styling copied from mean_literacy_by_strata.py
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title(title_map[col], fontsize=14, pad=10)
    
    # Remove top and right box borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Increase the thickness of the axis lines
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Grid lines behind bars
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_legend().remove()

# Add global legend and Y-axis label
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.02))
fig.supylabel('Weighted Percentage (%)', fontsize=14, x=0.01)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12, left=0.08, wspace=0.2, hspace=0.3)

plot1_path = os.path.join(output_dir, 'rq1_1_demographics_by_track.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"Plot 1 saved to: {plot1_path}")

# -----------------------------------------------------------------------------
# 6. PLOTTING 1.2: INTERSECTIONAL COMPOSITION (STYLED)
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 8))

top_strata = df_results_1_2.groupby('intersectional_strata')['Weighted %'].mean().sort_values(ascending=False).head(10).index
plot_data_1_2 = df_results_1_2[df_results_1_2['intersectional_strata'].isin(top_strata)]

ax_inter = sns.barplot(
    data=plot_data_1_2,
    y='intersectional_strata',
    x='Weighted %',
    hue='Track',
    palette=colors,
    edgecolor='black',
    linewidth=1,
    order=top_strata,
    zorder=3
)

plt.tick_params(axis='both', labelsize=14)
plt.title('Top 10 Intersectional Strata Across Tracks\n(Gender - Migration - SES)', fontsize=14, pad=10)
plt.xlabel('Weighted Percentage (%)', fontsize=14)
plt.ylabel('')

ax_inter.spines['top'].set_visible(False)
ax_inter.spines['right'].set_visible(False)
ax_inter.spines['bottom'].set_linewidth(2)
ax_inter.spines['left'].set_linewidth(2)
plt.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)

plt.legend(title="", fontsize=12, loc='lower right')
plt.tight_layout()

plot2_path = os.path.join(output_dir, 'rq1_2_intersectional_composition.png')
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"Plot 2 saved to: {plot2_path}")

print("All scripts updated with targeted styling.")
