import os, sys
import pandas as pd
import numpy as np
import pickle 
from scipy.stats import norm  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETUP ---
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\regression'

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
    
    # Use VET_TC1 as the independent variable
    variables = ['VET_TC1', current_score_column]
    df_clean = subdataset_df.dropna(subset=variables)

    education = df_clean['VET_TC1'].values.reshape(-1, 1)
    score = df_clean[current_score_column].values.reshape(-1, 1)
    w0 = df_clean['SPFWT0']

    # Model fitting
    model = LinearRegression()
    model.fit(X=education, y=score, sample_weight=w0)

    original_coeff = model.coef_[0][0]
    original_intercept = model.intercept_[0]

    coeff_list.append(original_coeff)
    intercept_list.append(original_intercept)

    # Replicate weights for sampling variance
    replicate_coeff = []
    for j in range(1, 81):
        current_replicate_weight_column = 'SPFWT' + str(j)
        temporary_weights = df_clean[current_replicate_weight_column]
        modeltemp = LinearRegression()
        modeltemp.fit(X=education, y=score, sample_weight=temporary_weights)
        replicate_coeff.append(modeltemp.coef_[0][0])

    sampling_variance = ((np.array(replicate_coeff) - original_coeff)**2).sum() * variance_factor
    sampling_variance_list.append(sampling_variance)

# --- FINAL AGGREGATION ---
average_coeff = np.mean(coeff_list)
average_inter = np.mean(intercept_list)
average_variance = np.mean(sampling_variance_list)
imputation_var = np.array(coeff_list).var() * (1 + (1/10))
final_se = np.sqrt(average_variance + imputation_var)

summary = pd.DataFrame({
    'Variable': ['Vocational (VET_TC1)'],
    'Coefficient (Gap)': [average_coeff],
    'Std.Error': [final_se],
    't-stat': [average_coeff / final_se],
    'p-value': [(1 - norm.cdf(np.abs(average_coeff / final_se))) * 2]
})

print("\n--- FINAL REGRESSION RESULTS (VET_TC1) ---")
print(summary)

# --- PLOTTING ---

# Prepare labels and means
df_clean['VET_Label'] = df_clean['VET_TC1'].map({0: "Non-Vocational", 1: "Vocational"})
pv_cols = ['PVLIT' + str(i) for i in range(1, 11)]
df_clean['Mean_PVLIT'] = df_clean[pv_cols].mean(axis=1)

plt.figure(figsize=(10, 7))
sns.set_palette("Dark2")

# Create Violin Plot
ax = sns.violinplot(
    x='VET_Label', 
    y='Mean_PVLIT', 
    data=df_clean, 
    order=["Non-Vocational", "Vocational"],
    inner='quartile', 
    alpha=0.8
)

# Overlay Regression Line
# For a binary variable, we plot from x=0 to x=1
x_indices = [0, 1]
# Point at 0 is Intercept; Point at 1 is Intercept + Coefficient
y_line = [average_inter, average_inter + average_coeff]

plt.plot(x_indices, y_line, color='#e74c3c', marker='o', linewidth=4, 
         label=f'Difference: {average_coeff:.2f} pts', zorder=10)

# Aesthetics
plt.title('Literacy Skills by Vocational Orientation (VET_TC1) in Norway', fontsize=14, fontweight='bold')
plt.xlabel('Secondary Educational Orientation', fontsize=12)
plt.ylabel('Literacy Score (Mean PV)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(outputfolder, 'norway_vet_tc1_literacy.png'), dpi=300)
plt.show()