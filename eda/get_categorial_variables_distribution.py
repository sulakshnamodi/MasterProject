import pandas as pd
import matplotlib.pyplot as plt
import os

# NOTE: Replace 'norway_categorical_distribution_WITH_CODES.csv' 
# with the actual filename of your categorical report.
report_filename = 'norway_categorical_distribution_WITH_CODES.csv' 
output_folder = 'Categorical_Plots'

# --- 1. Load the Categorical Summary Report ---
# We assume the report structure matches the attached image (Variable, Code, Count, Proportion).
distribution_df = pd.read_csv(report_filename)

# Create an output folder for the charts
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 2. Loop Through All Variables and Plot ---
# Group the DataFrame by the 'Variable' column
grouped_variables = distribution_df.groupby('Variable')

print(f"Generating {len(grouped_variables)} charts...")

for variable_name, group_data in grouped_variables:
    # Filter out variables that have no counts or only missing values (if necessary)
    if group_data['Count'].sum() == 0:
        continue

    # Prepare data for plotting (Codes vs. Counts)
    plot_data = group_data.sort_values(by='Count', ascending=False)
    
    plt.figure(figsize=(8, 5))
    
    # Create the bar chart
    bars = plt.bar(
        plot_data['Code'].astype(str),  # Ensure codes are treated as labels
        plot_data['Count'],
        color='teal'
    )
    
    # Add proportions as text labels above the bars
    for bar in bars:
        height = bar.get_height()
        # Find the corresponding proportion for the bar height
        proportion = plot_data[plot_data['Count'] == height]['Proportion (%)'].iloc[0]
        plt.text(
            bar.get_x() + bar.get_width() / 2., 
            height + 50, 
            f'{proportion:.1f}%',
            ha='center', 
            va='bottom', 
            fontsize=9
        )

    # Set titles and labels
    plt.title(f'Distribution of {variable_name}', fontsize=14)
    plt.xlabel('Response Codes (including missing flags)', fontsize=10)
    plt.ylabel('Frequency (Count)', fontsize=10)
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the chart to the designated folder
    plt.savefig(os.path.join(output_folder, f'Distribution_{variable_name}.png'))
    plt.close() # Close the figure to free up memory

print(f"\n✅ All charts have been generated and saved to the '{output_folder}' folder.")