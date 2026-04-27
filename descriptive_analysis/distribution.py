import pandas as pd
import pickle
import os
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. SETUP AND DATA LOADING
# -----------------------------------------------------------------------------
# Define the path to the preprocessed sub-dataset
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filepath = os.path.join(subdataset_root, 'piaac_norway_subdataset1.pkl')

# Load the pickle file containing the dataset
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

# Extract the working dataframe. Using .copy() ensures we don't modify the original loaded object.
df = loaded_data['dataframe'].copy()

# -----------------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS (SAMPLE DEMOGRAPHICS)
# -----------------------------------------------------------------------------
# Define a mapping dictionary to translate raw column names and numeric codes 
# into readable labels for our demographic tables.
table_config = {
    'EDUCATION TRACK': {'col': 'ED_GROUP', 'vals': {0.0: 'Academic', 1.0: 'Vocational'}},
    'GENDER':          {'col': 'GENDER_R', 'vals': {1.0: 'Male', 2.0: 'Female'}},
    'MIGRATION':       {'col': 'A2_Q03a_T', 'vals': {1.0: 'Native-born', 2.0: 'Foreign-born'}},
    'SES':             {'col': 'PAREDC2',   'vals': {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'}}
}

print(f"Total Analytical Sample (N): {len(df)}")
print("-" * 50)

print("--- TABLE 1: DESCRIPTIVES ---")
# Calculate the total sum of survey weights (SPFWT0) to compute weighted percentages
total_weight = df['SPFWT0'].sum()

# Iterate through the configuration to print the unweighted count (N) 
# and the weighted percentage (%) for each demographic subgroup.
for label, info in table_config.items():
    col = info['col']
    for val, cat_name in info['vals'].items():
        subset = df[df[col] == val]
        
        # Unweighted count of individuals in this subgroup
        n_count = len(subset)
        
        # Weighted percentage: (Sum of weights in subgroup / Total sum of weights) * 100
        pct = (subset['SPFWT0'].sum() / total_weight) * 100
        
        # Print formatted output
        print(f"{label:<15} | {cat_name:<10} | N={n_count:<5} | %={pct:.2f}")


# -----------------------------------------------------------------------------
# 3. MEAN LITERACY PROFICIENCY BY DEMOGRAPHIC STRATA
# -----------------------------------------------------------------------------
# PIAAC provides 10 "Plausible Values" (PVs) for literacy. 
# We average them to get a single, robust literacy score for each individual.
pv_cols = [f'PVLIT{i}' for i in range(1, 11)]
df['AVG_LIT'] = df[pv_cols].mean(axis=1)

# Define the variables and their categories to summarize mean literacy for
strata_vars = {
    'ED_GROUP': {'University': 0.0, 'Vocational': 1.0},
    'GENDER_R': {'Female': 2.0, 'Male': 1.0},
    'PAREDC2': {'High': 3.0, 'Medium': 2.0, 'Low': 1.0},
    'A2_Q03a_T': {'Native': 1.0, 'Foreign-born': 2.0}
}

clean_data = []
# Calculate the survey-weighted mean literacy score for each subgroup
for demo, subgroups in strata_vars.items():
    for sub_name, val in subgroups.items():
        subset = df[df[demo] == val]
        if not subset.empty:
            # Formula for weighted mean: sum(value * weight) / sum(weight)
            w_mean = (subset['AVG_LIT'] * subset['SPFWT0']).sum() / subset['SPFWT0'].sum()
            clean_data.append({
                'Demographic': demo.replace('ED_GROUP', 'Education Track')
                                   .replace('GENDER_R', 'Gender')
                                   .replace('PAREDC2', 'Parental SES')
                                   .replace('A2_Q03a_T', 'Migration'),
                'Subgroup': sub_name,
                'Mean Literacy Score': round(w_mean, 2)
            })

# Convert the results into a clean DataFrame for display
clean_df = pd.DataFrame(clean_data)
print("\n--- MEAN LITERACY PROFICIENCY BY STRATA ---")
print(clean_df.to_string(index=False))


# -----------------------------------------------------------------------------
# 4. STATISTICAL SIGNIFICANCE TESTING (WEIGHTED LEAST SQUARES)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("STATISTICAL SIGNIFICANCE AND STANDARD ERRORS")
print("="*60)

# Define variables to test against the mean literacy score
test_vars = {
    'ED_GROUP': 'Education Track',
    'GENDER_R': 'Gender',
    'PAREDC2': 'Parental SES',
    'A2_Q03a_T': 'Migration'
}

# Run a Weighted Least Squares (WLS) regression for each variable.
# This accounts for the complex survey design (via weights) when testing for group differences.
for var, name in test_vars.items():
    print(f"\n--- Testing: {name} ---")
    
    # Fit the WLS model. 'C()' indicates the independent variable is categorical.
    model = smf.wls(f"AVG_LIT ~ C({var})", data=df, weights=df['SPFWT0']).fit()
    
    # Extract and print the relevant table containing Coefficients, Standard Errors, and P-values
    res = model.summary2().tables[1] 
    print(res[['Coef.', 'Std.Err.', 'P>|t|']])


# -----------------------------------------------------------------------------
# 5. CORRELATION ANALYSIS AND HEATMAP
# -----------------------------------------------------------------------------
# Select variables of interest for correlation analysis
corr_cols = [
    'AVG_LIT',               # Average Literacy Score
    'READWORKC2_WLE_CA_T1',  # Work Reading Skills
    'WRITWORKC2_WLE_CA',     # Work Writing Skills
    'PAREDC2',               # Parental SES
    'A2_Q03a_T'              # Migration Status
]

# Create a subset and rename columns for clearer output labels
df_corr = df[corr_cols].rename(columns={
    'AVG_LIT': 'Literacy Score',
    'READWORKC2_WLE_CA_T1': 'Work Reading',
    'WRITWORKC2_WLE_CA': 'Work Writing',
    'PAREDC2': 'Parental SES',
    'A2_Q03a_T': 'Migration'
})

# Calculate the Pearson correlation coefficient matrix
correlation_matrix = df_corr.corr(method='pearson')

print("\n--- PEARSON CORRELATION COEFFICIENT (r) TABLE ---")
print(correlation_matrix.round(3))

# Plot and display the correlation matrix as a visual heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Correlation Heatmap: Literacy and Workplace Factors')
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# 6. METHODOLOGICAL INTEGRITY CHECK (ATTRITION BIAS ANALYSIS)
# -----------------------------------------------------------------------------
# Here we check if the data cleaning process (e.g., dropping missing values)
# disproportionately removed specific demographic groups, introducing bias.

# 'df_original' represents the data BEFORE cleaning
# 'df_final' represents the analytical sample AFTER cleaning
df_original = loaded_data['dataframe'].copy()
df_final = df.copy() 

cols_to_check = ['ED_GROUP', 'GENDER_R', 'PAREDC2', 'A2_Q03a_T']
bias_results = []

for col in cols_to_check:
    # Calculate the weighted proportions for each group in the ORIGINAL dataset
    orig_prop = df_original.groupby(col)['SPFWT0'].sum() / df_original['SPFWT0'].sum()
    
    # Calculate the weighted proportions for each group in the FINAL dataset
    final_prop = df_final.groupby(col)['SPFWT0'].sum() / df_final['SPFWT0'].sum()
    
    # Compare the proportions to see the difference (bias)
    comp = pd.DataFrame({'Original': orig_prop, 'Final': final_prop})
    comp['Difference'] = comp['Final'] - comp['Original']
    comp['Variable'] = col
    bias_results.append(comp)

# Concatenate and print the comparison results
comparison_table = pd.concat(bias_results)
print("\n--- METHODOLOGICAL INTEGRITY CHECK (Proportion Comparison) ---")
print(comparison_table.round(4))
