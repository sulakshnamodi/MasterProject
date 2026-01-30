import os, sys
import pandas as pd
import numpy as np
import pickle 
from scipy.stats import norm  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. SETUP & DATA LOADING ---
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\multiple-regression'

if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

subdataset_df = loaded_data['dataframe']
independent_vars = ['EDCAT6_TC1', 'WRITWORKC2_WLE_CA', 'WRITHOMEC2_WLE_CA', 'READWORKC2_WLE_CA_T1', 'READHOMEC2_WLE_CA_T1']


# --- 2. PRE-REGRESSION DIAGNOSTICS ---
# We use a single cleaned set for Correlation and VIF validation
df_val = subdataset_df.dropna(subset=independent_vars + ['PVLIT1']).copy()
scaler_val = StandardScaler()
X_val_scaled = scaler_val.fit_transform(df_val[independent_vars])

# A. Correlation Heatmap
print("\n--- GENERATING CORRELATION MATRIX ---")
plt.figure(figsize=(10, 8))
sns.heatmap(df_val[independent_vars].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix (Norway)')
plt.savefig(os.path.join(outputfolder, 'norway_correlation_heatmap.png'), dpi=300)
plt.show()

# B. VIF Calculation (This fixes the 'X_scaled' error)
X_vif_df = pd.DataFrame(X_val_scaled, columns=independent_vars)
X_vif_df['intercept'] = 1
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif_df.values, i) for i in range(X_vif_df.shape[1])]
print("\n--- VARIANCE INFLATION FACTOR (VIF) ---")
print(vif_data[vif_data['Variable'] != 'intercept'])

# --- 3. MULTIPLE REGRESSION WITH PVs & REPLICATE WEIGHTS ---
coeff_list, intercept_list, sampling_variance_list = [], [], []
fay_factor = 0.3 
variance_factor = 1 / (80 * (1 - fay_factor)**2) 
scaler = StandardScaler()

for i in range(1, 11):
    current_score = 'PVLIT' + str(i)
    df_loop = subdataset_df.dropna(subset=independent_vars + [current_score]).copy()
    
    # Standardizing inside the loop for each PV
    X_scaled = scaler.fit_transform(df_loop[independent_vars].values)
    y = df_loop[current_score].values.reshape(-1, 1)
    
    # Main Model
    model = LinearRegression().fit(X_scaled, y, sample_weight=df_loop['SPFWT0'])
    coeff_list.append(model.coef_.flatten())
    intercept_list.append(model.intercept_)

    # Replicate Weights (This fixes the 'X' squiggle)
    replicate_coeff = []
    for j in range(1, 81):
        w_rep = df_loop['SPFWT' + str(j)]
        rep_model = LinearRegression().fit(X_scaled, y, sample_weight=w_rep)
        replicate_coeff.append(rep_model.coef_.flatten())

    diffs = np.array(replicate_coeff) - model.coef_.flatten()
    sampling_variance_list.append((diffs**2).sum(axis=0) * variance_factor)

# --- 4. FINAL RESULTS & PLOTTING ---
avg_coeffs = np.mean(coeff_list, axis=0)
avg_samp_var = np.mean(sampling_variance_list, axis=0)
imputation_var = np.var(coeff_list, axis=0) * (1 + (1/10))
final_se = np.sqrt(avg_samp_var + imputation_var)

summary = pd.DataFrame({
    'Variable': independent_vars,
    'Coefficient': avg_coeffs,
    'Std.Error': final_se,
    't-stat': avg_coeffs / final_se,
    'p-value': [(1 - norm.cdf(np.abs(stat))) * 2 for stat in (avg_coeffs / final_se)]
})
print("\n--- FINAL REGRESSION RESULTS ---")
print(summary)

# --- 5. FOREST PLOT VISUALIZATION ---

# Calculate 95% Confidence Intervals (1.96 * SE)
summary['CI_lower'] = summary['Coefficient'] - (1.96 * summary['Std.Error'])
summary['CI_upper'] = summary['Coefficient'] + (1.96 * summary['Std.Error'])

# Define readable labels for the plot
label_map = {
    'EDCAT6_TC1': 'Highest Education Level',
    'WRITWORKC2_WLE_CA': 'Writing at Work (Index)',
    'WRITHOMEC2_WLE_CA': 'Writing at Home (Index)',
    'READWORKC2_WLE_CA_T1': 'Reading at Work (Index)',
    'READHOMEC2_WLE_CA_T1': 'Reading at Home (Index)'
}
summary['Variable_Label'] = summary['Variable'].map(label_map)

# Set the plot style
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Sort by coefficient value to make the plot easier to read
summary_sorted = summary.sort_values(by='Coefficient', ascending=True)

# Plot the error bars (Confidence Intervals)
plt.hlines(y=range(len(summary_sorted)), 
           xmin=summary_sorted['CI_lower'], 
           xmax=summary_sorted['CI_upper'], 
           color='#2c3e50', alpha=0.7, linewidth=2.5)

# Plot the point estimates (Coefficients)
# We color points based on significance (p < 0.05)
colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in summary_sorted['p-value']]
plt.scatter(summary_sorted['Coefficient'], 
            range(len(summary_sorted)), 
            color=colors, s=120, edgecolors='white', zorder=3, label='Standardized Coefficient')

# Add a vertical line at zero (Significance threshold)
plt.axvline(x=0, color='#34495e', linestyle='--', linewidth=1.5)

# Aesthetics and Labels
plt.yticks(range(len(summary_sorted)), summary_sorted['Variable_Label'], fontsize=12)
plt.xlabel('Standardized Coefficient (Beta) with 95% CI', fontsize=13, labelpad=15)
plt.title('Predictors of Literacy Skills in Norway\n(Standardized Multiple Regression)', 
          fontsize=16, fontweight='bold', pad=20)

# Add significance annotations next to the dots
for i, (coeff, p) in enumerate(zip(summary_sorted['Coefficient'], summary_sorted['p-value'])):
    sig_star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    plt.text(coeff, i + 0.15, sig_star, ha='center', fontweight='bold', color='#2c3e50')

# Save and Show
plt.tight_layout()
forest_plot_path = os.path.join(outputfolder, 'norway_literacy_forest_plot.png')
plt.savefig(forest_plot_path, dpi=300)
plt.show()

print(f"\nForest plot saved successfully to: {forest_plot_path}")