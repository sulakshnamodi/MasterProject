import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


data_filename = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\prgnorp2.csv'
report_filename = 'norway_numerical_distribution_RAW.csv' 
output_folder = 'Numerical_Plots'

# --- 1. Load Data and Report (CORRECTED LOADING) ---
try:
    # Load the full raw dataset
    raw_df = pd.read_csv(data_filename, sep=';', low_memory=False)
    
    # FIX: Load the summary report and tell pandas to use the first column (index 0) 
    # as the row index, which contains the variable names.
    report_df = pd.read_csv(report_filename, index_col=0) 
    
except FileNotFoundError:
    print("Error: Ensure both 'prgnorp2.csv' and the numerical report CSV file are in the correct directory.")
    exit()

# FIX: Extract the list of numerical columns directly from the clean index
numerical_cols = report_df.index.tolist()

# Create an output folder for the charts
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 2. Loop Through All Numerical Variables and Plot (CLEANED) ---

print(f"Generating charts for {len(numerical_cols)} potential numerical variables...")

# Define common survey missing codes
missing_codes = ['.', -9, -99, -999, 'DK', 'RF', 'NA', 'NI', 'M']

for variable_name in numerical_cols:
    # No need for the 'if variable_name in [...]' filter here, as the index extraction is clean.
        
    try:
        # --- Data Cleaning for Plotting (Local and temporary) ---
        # 1. Replace special codes with NaN and convert to float
        data_to_plot = raw_df[variable_name].replace(missing_codes, np.nan).astype(float)
        data_to_plot = data_to_plot.dropna() # Drop actual NaN values for plotting
        
        # Skip variables with too few data points remaining
        if len(data_to_plot) < 100:
            continue

        # --- Create Figure with Subplots (Box Plot and Histogram) ---
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [1, 4]})
        fig.suptitle(f'Distribution of {variable_name}', fontsize=16)

        # 1. Box Plot (Top Subplot)
        axes[0].boxplot(data_to_plot, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axes[0].set_yticks([]) 
        axes[0].set_title('Box Plot (Outliers and Quartiles)', fontsize=10)

        # 2. Histogram (Bottom Subplot)
        axes[1].hist(data_to_plot, bins=50, color='teal', edgecolor='black')
        axes[1].set_xlabel(f'Values of {variable_name}')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        
        # Save the chart
        plt.savefig(os.path.join(output_folder, f'Distribution_{variable_name}.png'))
        plt.close() 
        
    except KeyError:
        # Catch the error if the variable name is still not found (e.g., if it's a statistic like 'mean')
        print(f"Skipping: '{variable_name}' not found as a column in the raw dataset.")
    except Exception as e:
        # Catch any other unexpected error 
        print(f"An unexpected error occurred while processing {variable_name}: {e}")

print(f"\n✅ All numerical charts have been generated and saved to the '{output_folder}' folder.")