import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
import support as su

# Norway cleaned data
data_folder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed'
input_filename = 'norway_data_cleaned.csv'
input_filepath = os.path.join(data_folder, input_filename)
norway_df = pd.read_csv(input_filepath)

# Codebook
metadata_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\metadata'
codebook_filename = 'piaac-cy2-international-codebook.xlsx'
codebook_filepath = os.path.join(metadata_root, codebook_filename)
codebook_df = pd.read_excel(codebook_filepath)

plotting_folder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\distribution'

# Process each column and plot its distribution
# 1st we have to determine the variable type:
# As categorial/discreate variables: Bar Plot or Count Plot and 
# Numeric variable: Histogram or Kernel Density Estimate

df = norway_df

# Ensures that directory exists, if not, it creates it.
os.makedirs(plotting_folder, exist_ok=True)

# Get a list of all column names
# Refers to a variable 'columns' internally storing all the column names for this dataframe
columnlist = df.columns


# Go over each column in a dataframe and plot the distribution
# enumerate gives column_number and column_name for each element in the list
for col_number, col_name in enumerate(columnlist):
    
 
    plt.figure()
    # For Numerical Variables
    # Counts the number of unique values in a column
    if len(df[col_name].unique()) > 15:
        sns.histplot( data=df, x=col_name,  bins=100)
    
    # For Categorical Variables
    else:
        col_dict = su.get_mapping_dictionary(column_name=col_name, codebook_df=codebook_df)
        print(col_dict)
        
        if col_dict is None:
            print(f"Skipping plot for '{col_name}' because no mapping was found in the codebook.")
            plt.close() # Important to close the figure we opened earlier
            continue # Skip to the next column in the loop
    
        modified_column = su.replace_column_values(column_data=df[col_name], col_dict=col_dict)
       
        # Set order for bars, actual data as first and missing as last bar
        # 1. Get all unique labels from the modified column
        # This list will naturally include all descriptive labels AND the 'Missing' string.
        all_labels = modified_column.unique().tolist()

        # 2. Separate the 'Missing' label from the data labels
        # We use a conditional list comprehension to filter the list.
        data_labels = [label for label in all_labels if label != 'Missing']

        # 3. Sort the data labels (optional but good practice)
        # Sorting ensures categories like 'No' come before 'Yes' if they share the same starting code,
        # or puts them in alphabetical order.
        data_labels.sort() 

        # 4. Create the final order list, forcing 'Missing' to the end
        final_order = data_labels + ['Missing']
        # Set up the plotting style
       
        sns.countplot( x=modified_column, palette='Dark2', order=final_order )
        # sns.countplot( x=modified_column, palette='Dark2' )
        plt.xticks(rotation=30 )

    # Finalize the plot
    plt.title(f'{col_name} Distribution')
    plt.tight_layout()
    
    
    new_col_name = col_name.replace('/', '_').replace(':', '-').replace(' ', '_').replace('(', '').replace(')', '')

    # Choose a descriptive filename and extension
    filename = f'{new_col_name}_distribution.png'
    full_path = os.path.join(plotting_folder, filename)
    
    

    # Save the plot to a specific path
    plt.savefig(full_path)

    plt.show()

    # Close the plot figure
    plt.close()
    print(f"Plot saved successfully to: {full_path}")
    
    







