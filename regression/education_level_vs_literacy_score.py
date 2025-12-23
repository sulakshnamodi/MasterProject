import os, sys
import pandas as pd
import numpy as np
import pickle 
from scipy.stats import norm  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# RQ1: How do the levels of cognitive skills (literacy, numeracy, problem-solving) compare between university and vocationally educated adults at work in Norway?

# Load the Sub-dataset
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\regression'

with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

subdataset_df = loaded_data['dataframe']
metadata = loaded_data['metadata']
 
coeff_list = []
intercept_list = []
sampling_variance_list = []

fay_factor = 0.3 # Fay's Method, down weights a group of people by 30% and up-weights the weights
variance_factor = 1 / (80 * (1 - fay_factor)**2) 

# Iterate over each plausible value
for i in range(1,11):

    # Name of each PV of literacy score
    current_score_column = 'PVLIT' + str(i)

    # Removing the rows with missing values
    variables = ['EDCAT6_TC1', current_score_column]
    df_clean = subdataset_df.dropna(subset=variables)

    education = df_clean['EDCAT6_TC1'].values.reshape(-1,1)
    score = df_clean[current_score_column].values.reshape(-1,1)
    w0 = df_clean['SPFWT0']

    # Initialising  and fitting the model
    model = LinearRegression()
    model.fit(X=education, y=score, sample_weight=w0)

    original_coeff = model.coef_
    original_intercept = model.intercept_

    # View the values of m and c
    print(f"Coefficients: {original_coeff}")
    print(f"Intercept: {original_intercept}")

    # Transferring coefficient to the list for future use
    coeff_list.append(original_coeff)
    intercept_list.append(original_intercept)


    replicate_coeff = []

    # Use the same score and education , but calculate regression using 80 different weights
    for j in range(1, 81):

        current_replicate_weight_column = 'SPFWT' + str(j)

        temporary_weights = df_clean[current_replicate_weight_column]

        modeltemp = LinearRegression()
        modeltemp.fit(X=education, y=score, sample_weight=temporary_weights)

        replicate_coeff.append(modeltemp.coef_)

    
    sampling_variance =  ((replicate_coeff - original_coeff)**2).sum() * variance_factor
    print(sampling_variance)
    sampling_variance_list.append(sampling_variance)


average_coeff = np.mean(coeff_list)
average_inter = np.mean(intercept_list)
average_variance = np.mean(sampling_variance_list)

print(f"the average coeff is: {average_coeff}")
print(f"the average intercept is: {average_inter}")

# Need to understand better later! 
imputation_var = np.array(coeff_list).var() * (1 + (1/10))
final_se = np.sqrt(average_variance + imputation_var)



summary = pd.DataFrame({
    'Coefficient': [average_coeff],
    'Std.Error': [final_se],
    't-stat': [average_coeff / final_se],
    # Use norm.cdf from scipy.stats directly
    'p-value': [(1 - norm.cdf(np.abs(average_coeff / final_se))) * 2]
})

print("\n--- FINAL REGRESSION RESULTS (NORWAY) ---")
print(summary)



# --- PLOTTING ---

# 1. Map education codes to descriptive labels
edu_labels = {
    1: "Lower secondary",
    2: "Upper secondary",
    3: "Post-secondary",
    4: "Tertiary professional",
    5: "Tertiary bachelor",
    6: "Tertiary master/research"
}

# Create a display column for the x-axis
df_clean['Edu_Label'] = df_clean['EDCAT6_TC1'].map(edu_labels)
unique_labels = [edu_labels[i] for i in range(1, 7) if i in edu_labels]

# Prepare literacy mean for visualization
pv_cols = ['PVLIT' + str(i) for i in range(1, 11)]
df_clean['Mean_PVLIT'] = df_clean[pv_cols].mean(axis=1)

# 2. Setup plotting style
plt.figure(figsize=(14, 8))
sns.set_palette("Dark2") # Using Dark2 color palette

# 3. Create the Violin Plot
ax = sns.violinplot(
    x='Edu_Label', 
    y='Mean_PVLIT', 
    data=df_clean, 
    order=unique_labels, 
    inner='quartile', 
    alpha=0.8
)

# 4. Overlay the Regression Line
# We use indices (0 to 5) for the x-coordinates so the line aligns with the violins
x_indices = np.arange(len(unique_labels))
# Re-calculate y based on the original numeric values (1 to 6)
y_line = average_inter + average_coeff * np.arange(1, 7)

plt.plot(x_indices, y_line, color='#e74c3c', linewidth=4, 
         label=f'Regression Trend (Slope: {average_coeff:.2f})', zorder=10)

# 5. Aesthetics and Formatting
plt.title('Cognitive Literacy Skills by Education Level in Norway', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Educational Attainment', fontsize=13, labelpad=15)
plt.ylabel('Literacy Score (Mean of Plausible Values)', fontsize=13)

# Rotating labels for better readability
plt.xticks(rotation=30, ha='right')

# Remove top and right axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add horizontal gridlines for easier score reading
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.legend(frameon=False, loc='upper left', fontsize=11)

# 6. Save and Show
plt.tight_layout()
filepath = os.path.join(outputfolder, 'norway_literacy_education_violin.png')
plt.savefig(filepath, dpi=300)
print("\nViolin plot with Dark2 palette and labels saved successfully.")