import os, sys
import pandas as pd
import numpy as np
import pickle 
from scipy.stats import norm  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS ---
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\regression'

# Load data
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

subdataset_df = loaded_data['dataframe']

coeff_list = []
intercept_list = []
sampling_variance_list = []

fay_factor = 0.3 
variance_factor = 1 / (80 * (1 - fay_factor)**2) 

# --- REGRESSION LOOP ---
for i in range(1, 11):
    current_score_column = 'PVLIT' + str(i)
    
    # We use VETC2 as our binary predictor
    variables = ['VETC2', current_score_column]
    df_clean = subdataset_df.dropna(subset=variables)

    # 0 = No (Academic/University), 1 = Yes (Vocational)
    education = df_clean['VETC2'].values.reshape(-1, 1)
    score = df_clean[current_score_column].values.reshape(-1, 1)
    w0 = df_clean['SPFWT0']

    model = LinearRegression()
    model.fit(X=education, y=score, sample_weight=w0)

    original_coeff = model.coef_[0][0]
    original_intercept = model.intercept_[0]

    coeff_list.append(original_coeff)
    intercept_list.append(original_intercept)

    # Replicate Weights Calculation
    replicate_coeff = []
    for j in range(1, 81):
        current_replicate_weight_column = 'SPFWT' + str(j)
        temporary_weights = df_clean[current_replicate_weight_column]

        modeltemp = LinearRegression()
        modeltemp.fit(X=education, y=score, sample_weight=temporary_weights)
        replicate_coeff.append(modeltemp.coef_[0][0])

    sampling_variance = ((np.array(replicate_coeff) - original_coeff)**2).sum() * variance_factor
    sampling_variance_list.append(sampling_variance)

# Final Statistics
average_coeff = np.mean(coeff_list)
average_inter = np.mean(intercept_list)
average_variance = np.mean(sampling_variance_list)
imputation_var = np.array(coeff_list).var() * (1 + (1/10))
final_se = np.sqrt(average_variance + imputation_var)

summary = pd.DataFrame({
    'Variable': ['Vocational (VETC2)'],
    'Coefficient': [average_coeff],
    'Std.Error': [final_se],
    't-stat': [average_coeff / final_se],
    'p-value': [(1 - norm.cdf(np.abs(average_coeff / final_se))) * 2]
})

print("\n--- FINAL REGRESSION RESULTS (VETC2) ---")
print(summary)

# --- PLOTTING ---

# 1. Prepare Display Data
df_clean['Edu_Type'] = df_clean['VETC2'].map({0: 'Non-Vocational', 1: 'Vocational'})
pv_cols = ['PVLIT' + str(i) for i in range(1, 11)]
df_clean['Mean_PVLIT'] = df_clean[pv_cols].mean(axis=1)

plt.figure(figsize=(10, 7))
sns.set_palette("Dark2")

# 2. Create the Violin Plot
ax = sns.violinplot(
    x='Edu_Type', 
    y='Mean_PVLIT', 
    data=df_clean, 
    order=['Non-Vocational', 'Vocational'],
    inner='quartile', 
    alpha=0.7
)

# 3. Overlay the Regression Line (Fixed shapes for binary variable)
# x_indices are 0 and 1 (corresponding to the two violins)
x_indices = np.array([0, 1])
# y_points: Intercept (at x=0) and Intercept + Coeff (at x=1)
y_line = np.array([average_inter, average_inter + average_coeff])

plt.plot(x_indices, y_line, color='#e74c3c', marker='o', markersize=10, linewidth=4, 
         label=f'Difference: {average_coeff:.2f} points', zorder=10)

# 4. Aesthetics
plt.title('Literacy Score Gap: Non-Vocational vs. Vocational (Norway)', fontsize=15, fontweight='bold')
plt.ylabel('Literacy Score (Plausible Values Mean)')
plt.xlabel('Education Orientation')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(outputfolder, 'norway_vetc2_literacy_violin.png'), dpi=300)
plt.show()