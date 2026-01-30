import os, sys
import pandas as pd
import numpy as np
import pickle 
from scipy.stats import norm  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sn

# VETC2:Respondent’s highest level of education is vocationally oriented (derived)
# 0: No; 1: Yes

# setting of the subdataset
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\regression'

key_variable =  'ED_GROUP'

# load the subdataset (pkl file)
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

subdataset_df = loaded_data['dataframe']

# Create list to save values of coefficient, intercept and sampling weight
coeff_list = []
intercept_list = []
sampling_variance_list = []

# Fay's Method, down weights a group of people by 30% and up-weights the weights
fay_factor = 0.3 
variance_factor = 1 / (80 * (1 - fay_factor)**2) 

# Regression: Find the relationship between literacy score of vocational vs non-vocation higher studies
for i in range(1, 11):
    current_score_column = 'PVLIT' + str(i)

    # Drop rows with missing values in columns ED_GROUP and PVLIT1-PVLIT10
    variables = [key_variable, current_score_column]
    df_clean = subdataset_df.dropna(subset=variables)

    # Select the variables for regression
    highest_education = df_clean[key_variable].values.reshape(-1, 1)
    score = df_clean[current_score_column].values.reshape(-1, 1)
    w0 = df_clean['SPFWT0']

    # Select model and fit the model for regression
    model = LinearRegression()
    model.fit(X=highest_education, y=score, sample_weight=w0)

    # Use the getter model.coef_ and model.intercept_ to pull the values from each loop
    # Try to understand why we use the square braket with the 0 here
    original_coeff = model.coef_[0][0]
    original_intercept = model.intercept_[0]

    # Add the original coefficient in the empty list we created above
    coeff_list.append(original_coeff)
    intercept_list.append(original_intercept)

    # Create a second loop to loop through the replicate weight SPFWT1- SPFWT80
    # Use same PV score and highest education
    
    # Create an empty list for to save the sampling variance
    replicate_coeff = []

    for j in range(1, 81):
        current_replicate_weight_column = 'SPFWT' + str(j)
        temporary_weights = df_clean[current_replicate_weight_column]
    # Choose model, fit the model with the variables
        modeltemp = LinearRegression()
        modeltemp.fit(X=highest_education, y=score, sample_weight=temporary_weights)
        
        #Use the getter modeltemp.coeff_ to pull the coefficient
        original_replicate_coeff = modeltemp.coef_[0][0]
        # Add values to empty list
        replicate_coeff.append(original_replicate_coeff)

      #sampling variance: Uncertaity from people, refer to definition of fay's factor
    sampling_variance =  ((np.array(replicate_coeff) - original_coeff)**2).sum() * variance_factor
    print(sampling_variance)
    sampling_variance_list.append(sampling_variance)

# Calculating final statistics
average_coeff = np.mean(coeff_list)
average_inter = np.mean(intercept_list)
average_variance = np.mean(sampling_variance_list)
imputation_var = np.array(coeff_list).var() * (1 + (1/10))
final_se = np.sqrt(average_variance + imputation_var)

summary = pd.DataFrame({
    'Variable': ['Universit Vs Vocation (ED_GROUP)'],
    'Coefficient': [average_coeff],
    'Std.Error': [final_se],
    't-stat': [average_coeff / final_se],
    'p-value': [(1 - norm.cdf(np.abs(average_coeff / final_se))) * 2]
})

print(f"\n--- FINAL REGRESSION RESULTS ({key_variable}) ---")
print(summary)

# Plotting 


# 1. Prepare Display Data
df_clean['Edu_Type'] = df_clean[key_variable].map({0: 'University', 1: 'Vocational'})
pv_cols = ['PVLIT' + str(i) for i in range(1, 11)]
df_clean['Mean_PVLIT'] = df_clean[pv_cols].mean(axis=1)

plt.figure(figsize=(10, 7))
sn.set_palette("Dark2")

# 2. Create the Violin Plot
ax = sn.violinplot(
    x='Edu_Type', 
    y='Mean_PVLIT', 
    data=df_clean, 
    order=['University', 'Vocational'],
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
plt.title('Literacy Score Gap: University vs. Vocational in Norway', fontsize=15, fontweight='bold')
plt.ylabel('Literacy Score (Mean of Plausible Values)')
plt.xlabel('Highest Education Orientation')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(outputfolder, f'norway_{key_variable.lower()}_literacy_violin.png'), dpi=300)
plt.show()
