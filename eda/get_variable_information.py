import pandas as pd
import matplotlib.pyplot as plt
import os


meta_data_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\metadata'
missing_data_filename = 'piaac-cy2-missing_variables-puf.xlsx'
missing_data_filepath = os.path.join(meta_data_root, missing_data_filename)

# Load the data from the specific path into a Dataframe
missing_variables_df = pd.read_excel(missing_data_filepath)

# Filter for Norway Variables 
# Create a new dataframe to select only the rows where the 'NOR' column has the value 'Yes'
df_norway = missing_variables_df[missing_variables_df['NOR'] == 'Yes']


# Accesses the 'Group' column and get the unique groups from the filtered data 
# Use .unique() method to extract a NumPy array containing all distint, non-repated category names
unique_groups = df_norway['Group'].unique()

# A. List of Unique Groups

print("List of Unique Groups present in Norway data:")
for group in unique_groups:
    print(f"- {group}")

print("\n" + "="*50 + "\n")

# B. Variable Summary by Group (Using Pandas Idiomatic Method)

# Use .groupby().size() for efficient variable counting
group_summary = df_norway.groupby('Group').size().reset_index(
    name='Variable_Count_in_Norway'
)

# Sort the summary by count in descending order
group_summary = group_summary.sort_values(
    by='Variable_Count_in_Norway', 
    ascending=False
).reset_index(drop=True)

# Print the resulting summary
print("Variable Summary by Group (in Norway):")
print(group_summary)

# --- Generate the Visualization (Horizontal Bar Chart) ---
plt.figure(figsize=(10, 6))

# Create a horizontal bar chart
plt.barh(
    group_summary['Group'], 
    group_summary['Variable_Count_in_Norway'], 
    color='skyblue'
)

# Set labels and title
plt.xlabel('Variable Count in Norway')
plt.ylabel('Variable Group')
plt.title('Variable Count by Group (Norway Data)')

# Add grid lines for easier reading of values
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Invert y-axis so the largest group is at the top (standard for bar charts)
plt.gca().invert_yaxis()

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# Save the plot
plt.savefig('norway_variable_group_summary_barchart.png')
plt.show() # Use this in your VS Code environment to display the chart