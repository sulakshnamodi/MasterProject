import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Still needed for comparison/potential future use, but not for direct replacement
import os

# SETUP AND DATA LOADING 
nor_data_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data'
nor_data_filename = 'prgnorp2.csv'
nor_data_filepath = os.path.join(nor_data_root, nor_data_filename)

# Load the data, using the correct delimiter (;) and setting low_memory=False
# This DataFrame 'nor_data_df' retains all special codes (e.g., '.', -9) as their literal values.
nor_data_df = pd.read_csv(nor_data_filepath, sep=';', low_memory=False)

# --- 2. AUTOMATIC VARIABLE CATEGORIZATION (BASED ON CODE COUNT) ---

# Define a threshold: columns with few unique values are treated as categorical/discrete codes.
# This ensures survey codes (including missing codes) are grouped for the value_counts report.
UNIQUE_VALUE_THRESHOLD = 50 

numerical_cols = []
categorical_cols = []

print(f"Analyzing {len(nor_data_df.columns)} columns for type categorization...")

for col in nor_data_df.columns:
    # Use .nunique() on the column after converting it to string (object) 
    # to count unique string representations of codes/values.
    unique_count = nor_data_df[col].astype(str).nunique(dropna=True) 
    
    if unique_count <= UNIQUE_VALUE_THRESHOLD:
        # If the number of unique codes/values is small, treat as categorical (including missing codes)
        categorical_cols.append(col)
    else:
        # Otherwise, treat as numerical/continuous
        numerical_cols.append(col)

print(f"Categorized {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns.")

# --- 3. GENERATE AUTOMATED REPORTS ---

def report_numerical_distribution(df, cols, filename='norway_numerical_distribution_RAW.csv'):
    """
    Generates descriptive statistics for columns that are predominantly numerical.
    NOTE: Since special codes are not removed, this report may exclude columns 
    where pandas detects the presence of string codes like 'DK' or '.', 
    or it will exclude those codes from the calculations.
    """
    if not cols:
        print("\nNo numerical columns found to report.")
        return
        
    print(f"\nGenerating descriptive statistics report for {len(cols)} numerical columns...")
    
    # Use .describe() - pandas automatically filters out non-numeric codes when calculating stats
    numerical_summary = df[cols].describe().transpose()
    numerical_summary.to_csv(filename)
    print(f"✅ Numerical distribution saved to: {filename}")


def report_categorical_distribution(df, cols, filename='norway_categorical_distribution_WITH_CODES.csv'):
    """
    Generates frequency tables for all categorical columns, including missing codes.
    This report contains the distribution of ALL discrete codes (valid and missing).
    """
    if not cols:
        print("\nNo categorical columns found to report.")
        return
        
    print(f"\nGenerating frequency tables report for {len(cols)} categorical columns...")
    
    all_frequencies = []
    
    for col in cols:
        # Use .value_counts() on the original, un-cleaned data
        # dropna=False ensures any actual NaN values are counted (though unlikely in this raw scenario)
        counts = df[col].astype(str).value_counts(dropna=False)
        proportions = df[col].astype(str).value_counts(normalize=True, dropna=False).mul(100).round(2)
        
        # Combine into a temporary DataFrame
        temp_df = pd.DataFrame({
            'Variable': col,
            'Code': counts.index,
            'Count': counts.values,
            'Proportion (%)': proportions.values
        })
        all_frequencies.append(temp_df)
        
    # Concatenate all individual frequency tables into one large report
    if all_frequencies:
        categorical_summary = pd.concat(all_frequencies, ignore_index=True)
        categorical_summary.to_csv(filename, index=False)
        print(f"✅ Categorical distribution saved to: {filename}")


# --- 4. EXECUTION ---
report_numerical_distribution(nor_data_df, numerical_cols)
report_categorical_distribution(nor_data_df, categorical_cols)