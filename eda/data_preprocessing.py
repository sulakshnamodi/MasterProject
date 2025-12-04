# Data Preprocessing: 
# Loads PIAAC data and modifies column name 
# and saves the dataframe for reusing in future

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Norway data
norway_data_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\raw'
output_folder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed'
norway_data_filename = 'prgnorp2.csv'
norway_data_filepath = os.path.join(norway_data_root, norway_data_filename)

# Codebook
metadata_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\metadata'
codebook_filename = 'piaac-cy2-international-codebook.xlsx'
codebook_filepath = os.path.join(metadata_root, codebook_filename)

# Load the data, using the correct delimiter (;) and setting low_memory=False
# This DataFrame 'nor_data_df' retains all special codes (e.g., '.', -9) as their literal values.
norway_df = pd.read_csv(norway_data_filepath, sep=';', low_memory=False)
codebook_df = pd.read_excel(codebook_filepath)

print('Codebook Columns', codebook_df.columns)
print('Norway Data Columns', norway_df.columns)

# Rename column names in norway_df to readable values
## 1. Create the Column Mapping Dictionary
# We map the 'Variable' column (current names) to the 'Label' column (new names).

# It's safest to ensure unique 'Variable' names before creating the dictionary.
# If duplicates exist, this keeps the label from the *first* occurrence.
column_mapping_series = codebook_df.drop_duplicates(subset=['Variable'], keep='first').set_index('Variable')['Label']
column_mapping = column_mapping_series.to_dict()


## 2. Rename the Columns
# Use the .rename() method with the generated dictionary.
# Setting axis=1 or columns= specifies that the mapping is for column names.
# The rename operation will safely ignore any keys in the mapping that
# don't correspond to a column in norway_df.
norway_df = norway_df.rename(columns=column_mapping)

# Remove all the variables which as . values everywhere 
# Find columns where ALL values are the string '.'
# 1. Use the equality operator (==) to check every cell.
# 2. Use .all() to check if ALL values in the resulting boolean Series for that column are True.
# 3. Use .index to get the list of column names that meet this criteria.
cols_to_drop = norway_df.columns[
    (norway_df == '.').all()
].tolist()



# Drop the identified columns from the DataFrame
norway_df = norway_df.drop(columns=cols_to_drop)


norway_df = norway_df.replace('.', np.nan)
print("✅ Replaced '.' with np.nan across the entire DataFrame.")

# Iterate through columns and convert to numeric (float)
# The errors='coerce' argument will turn any remaining non-numeric strings (if any) into NaN,
# ensuring the final conversion to float is successful.
for col in norway_df.columns:
    # Check if the column is currently a string/object type that needs conversion
    if norway_df[col].dtype == 'object':
        
        # Attempt conversion. If the column is mostly non-numeric (e.g., names), 
        # this will result in a column full of NaNs, but it successfully changes the dtype.
        norway_df[col] = pd.to_numeric(norway_df[col], errors='coerce')

# After using pd.to_numeric, all resulting numeric columns are automatically float64.
print("✅ Converted all suitable columns to float.")



# Save the data frame as csv
output_filepath = os.path.join(output_folder, 'norway_data_cleaned.csv')
norway_df.to_csv(output_filepath, index=False)


