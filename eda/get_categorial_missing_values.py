import pandas as pd
import numpy as np
import os

# --- Configuration ---
REPORT_FILE = 'norway_categorical_distribution_WITH_CODES.csv'
MISSING_THRESHOLD = 50.0 # Percentage threshold for 'high' missing rate (e.g., 50%)
output_folder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results'


# Define known codes that represent missing values
# These codes must match the string representation used in your report (e.g., '-9' as a string)
MISSING_CODES = [
    '.', # Common placeholder for missing values
    'DK', 'RF', 'NA', 'NI', 'M', # Common string codes
    '-9', '-99', '-999', '-1', '-2', '-3' # Common integer codes (as strings)
]

# --- 1. Load the Categorical Distribution Report ---
try:
    report_df = pd.read_csv(REPORT_FILE)
except FileNotFoundError:
    print(f"Error: The report file '{REPORT_FILE}' was not found. Please ensure the file name is correct and accessible.")
    # In your local environment, you would ensure the path is correct here.
    exit()

# Ensure the 'Code' column is treated as a string for consistent matching
report_df['Code'] = report_df['Code'].astype(str).str.strip()

# --- 2. Filter for Missing Data ---
# Select only the rows corresponding to the defined missing codes
missing_data_df = report_df[report_df['Code'].isin(MISSING_CODES)].copy()

# --- 3. Group and Sum Proportions ---
# Sum the proportions of all missing codes for each variable
missing_summary = missing_data_df.groupby('Variable')['Proportion (%)'].sum().reset_index(name='Total_Missing_Proportion_(%)')

# --- 4. Identify High Missing Rate Variables ---
high_missing_variables = missing_summary[
    missing_summary['Total_Missing_Proportion_(%)'] >= MISSING_THRESHOLD
].sort_values(by='Total_Missing_Proportion_(%)', ascending=False)

# --- 5. Display and Save the Result ---
print(f"--- Variables with Total Missing Proportion >= {MISSING_THRESHOLD}% ---")
print(f"Total variables with high missing rate: {len(high_missing_variables)}")
print(high_missing_variables)

missing_categorial_filepath = os.path.join(output_folder, 'variables_high_missing_rate.csv')
high_missing_variables.to_csv(missing_categorial_filepath, index=False)
print(f"\nReport saved to: {missing_categorial_filepath}")