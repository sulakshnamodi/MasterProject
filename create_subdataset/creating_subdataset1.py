'''
File for creating a subdataset from PIAAC Norway.
Specifically contains: Literacy scores (all plausible values), 
gender, educational background, literacy use at work(readwork),


For research question 1)How do the levels of cognitive skills (literacy, numeracy,
problem-solving) compare between university and vocationally educated adults
at work in Norway?
'''
import os, sys
import pandas as pd
import numpy as np
import pickle


# Step1: Load the raw data file
norway_data_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\raw'
norway_data_filename = 'prgnorp2.csv'
norway_data_filepath = os.path.join(norway_data_root, norway_data_filename)
# Load the data, using the correct delimiter (;) and setting low_memory=False
# This DataFrame 'nor_data_df' retains all special codes (e.g., '.', -9) as their literal values.
norway_df = pd.read_csv(norway_data_filepath, sep=';', low_memory=False)

# Load the Codebook
metadata_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\metadata'
codebook_filename = 'piaac-cy2-international-codebook.xlsx'
codebook_filepath = os.path.join(metadata_root, codebook_filename)

codebook_df = pd.read_excel(codebook_filepath) 
print('Codebook Columns', codebook_df.columns)

# Creating an output folder
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'

# Ensures that the directory exits, if not, it create it
os.makedirs(outputfolder, exist_ok=True)

# Step2: Create a list of desired variables
'''
Population Variables
CNTRYID: Country ID, 578 is Norway
GENDER_R: Person gender
    1: Male; 2: Female
C2_D05:Current Employment status (devrived)
    1: Employed; 2: Unemployed; 3: Out of the labour force; 4: Not known 
PAIDWORK12: Adults who have had paid work during the 12 months preceding the survey (derived), 
    0: Has not had paid work during the 12 months preceding the survey; 1: Has had paid work during the 12 months preceding the survey
'''
background_vars = ['CNTRYID', 'GENDER_R', 'C2_D05', 'PAIDWORK12']


'''
Education Level Variables
EDCAT6_TC1 Highest level of formal education obtained (6 categories, ISCED 97) (derived),
    1: Lowe:r secondary or less (ISCED 1,2, 3C short or less); 2: Upper secondary (ISCED 3A-B, C long); 3: Post-secondary, non-tertiary (ISCED 4A-B-C); 4: Tertiary – professional degree (ISCED 5B); 5: Tertiary – bachelor degree (ISCED 5A); 6: Tertiary – master/research degree (ISCED 5A/6)
VET_TC1: Respondent’s upper secondary/post-secondary education is vocationally oriented (derived, Trend PIAAC 1/2)
     0: No; 1: Yes
VETC2:Respondent’s highest level of education is vocationally oriented (derived)
     0: No; 1: Yes
''' 
education_vars = ['EDCAT6_TC1', 'VET_TC1', 'VETC2']

'''
Cognitive Skills Variables
PVLIT (PVLIT1: Literacy scale score - Plausible value 1 to PVLIT10): 
Note: Taking all 10 values for valid statistical inference
'''
literacy_scores_vars = [f'PVLIT{i}' for i in range(1, 11)]

'''
Literacy Skills Usage Variables
WRITWORKC2_WLE_CA: Index of use of writing skills at work, categorised WLE (derived)
WRITHOMEC2_WLE_CA: Index of use of writing skills at home, categorised WLE (derived)
READWORKC2_WLE_CA_T1: Index of use of reading skills at work (prose and document texts), categorised WLE (derived) 
READHOMEC2_WLE_CA_T1: Index of use of reading skills at home (prose and document texts), categorised WLE (derived)
'''
literacy_usage_vars = ['WRITWORKC2_WLE_CA', 'WRITHOMEC2_WLE_CA', 'READWORKC2_WLE_CA_T1', 'READHOMEC2_WLE_CA_T1']


'''
Sampling and Replicate Weights 
SPFWT0 is the final full sample weight; SPFWT1-80 are replicate weights
'''
sampling_weight_vars = ['SPFWT0'] + [f'SPFWT{i}' for i in range(1, 81)]


'''
Work-Related Variables
ISCOSKIL4: Occupational classification of respondent's job (4 skill based categories), last or current (derived)
D2_Q04:Current work - Employee or self-employed
'''
work_vars = ['ISCOSKIL4', 'D2_Q04' ]


# Step3: Access the data of these variables from raw data file
all_variables = background_vars + education_vars + literacy_scores_vars + sampling_weight_vars + literacy_usage_vars + work_vars

# Step4: Verify the new dataframe
# Create a dataframe with all variables all_variables
all_variables_df = norway_df[all_variables]

# Step5: Clean and Preprocess the new dataframe
    # - Removing . columns, Nan Values
    # - Convert the numbers to float rather than strings
# Convert all variables in all_variable_df to float
# errors='coerce' turns non-numeric strings (like '.') into NaN
for col in all_variables:
    all_variables_df[col] = pd.to_numeric(all_variables_df[col], errors='coerce')

# Check the data types of all columns
# print(all_variables_df.dtypes)

#View the main weight and the first 5 replicate weights for the first person
print(all_variables_df[['SPFWT0', 'SPFWT1', 'SPFWT2', 'SPFWT3', 'SPFWT4', 'SPFWT5']].iloc[0])

# Step6: Save in to output formats (csv, pkl)
    # - For csv, modify column names to readable lables
    # - For pkl, save dataframe and other variables which map labels to variable names

# Create a dictionary for readable labels (mapping)

# Rename column names in all_variables_df to readable values
column_mapping_series = codebook_df.drop_duplicates(subset=['Variable'], keep='first').set_index('Variable')['Label']
column_mapping = column_mapping_series.to_dict()

# Save to CSV with readable headers
# We use .rename() to change headers for the export
output_csv_path = os.path.join(outputfolder, 'piaac_norway_subdataset1.csv')
all_variables_df.rename(columns=column_mapping).to_csv(output_csv_path, index=False)
print(f"Data saved to CSV: {output_csv_path}")

# Save to Pickle (pkl) as the data type is preserved as float
# Saving the dictionary containing both the dataframe and the mapping
description = """
Origin & Scope: This sub-dataset is generated from the PIAAC Cycle 2 (2023) International Survey, specifically filtered for the Norwegian target population (CNTRYID 578). 
Cognitive Skills: Includes 10 Plausible Values for Literacy (PVLIT1 to PVLIT10), used as the primary outcome measures for adult proficiency.
Education Background: Contains the highest level of formal education (EDCAT6) and vocational orientation flags (VET_TC1 or VET_TC2) to distinguish between academic and practical training.
Skill Use Indices: Includes Weighted Likelihood Estimates (WLE) for the frequency of reading and writing skills used both at work (READWORK, WRITWORK) and at home (READHOME, WRITHOME).
Labor Market Status: Captures the respondent’s current employment status (C_D05), occupational classification (ISCO1L or ISCOSKILL4), and work history.
Statistical Weights: Features the final full-sample weight (SPFWT0) and 80 replicate weights (SPFWT1 to SPFWT80) necessary for calculating accurate population estimates and standard errors.
Research Objective: This data is designed to examine cognitive skill gaps between university-educated and vocationally-educated adults within the Norwegian labor market.
"""

# To see it formatted exactly as above:
print(description)
output_pkl_path = os.path.join(outputfolder, 'piaac_norway_subdataset1.pkl')
saving_data = {
    'dataframe': all_variables_df,
    'metadata': column_mapping,
    'description': description
}


with open(output_pkl_path, 'wb') as fileobject:
    pickle.dump(saving_data, fileobject)
print(f"Data and labels saved to Pickle: {output_pkl_path}")

















