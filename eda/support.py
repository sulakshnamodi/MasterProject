import pandas as pd

# Assume the existence of the parse_scheme_string function from the previous context:
def parse_scheme_string(scheme_string):
    """Parses a scheme string (e.g., '1: Yes; -9: NA') into a dictionary."""
    if not isinstance(scheme_string, str):
        return {}
    
    mapping = {}
    pairs = scheme_string.split(';')

    for pair in pairs:
        if ':' in pair:
            try:
                code_str, label = pair.split(':', 1)
            except ValueError:
                continue

            code_str = code_str.strip()
            label = label.strip()

            # Attempt to convert code to integer
            try:
                code = int(code_str)
            except ValueError:
                # If conversion fails, keep it as a string
                code = code_str

            mapping[code] = label
    return mapping


def get_mapping_dictionary(column_name, codebook_df):
    """
    Retrieves and combines value mappings from three scheme columns 
    for a given variable from the codebook.
    
    Args:
        column_name (str): The 'Label' of the column to find (e.g., 'Gender of respondent').
        codebook_df (pd.DataFrame): The codebook DataFrame.

    Returns:
        dict or None: A comprehensive mapping dictionary, or None if the variable is not found.
    """
    
    # 1. Find the relevant row based on the 'Label'
    scheme_row = codebook_df.loc[
        codebook_df['Label'] == column_name,
        [
            'Value Scheme Detailed', 
            'Missing Scheme Detailed: SAS', 
            'Missing Scheme Detailed: SPSS'
        ]
    ]

    # Check if the variable exists
    if scheme_row.empty:
        print(f"⚠️ Warning: Variable with Label '{column_name}' not found in the codebook.")
        return None
    
    # Get the row data (should only be one row)
    schemes = scheme_row.iloc[0]
    
    # Initialize the final combined dictionary
    combined_dict = {}

    # 2. Iterate through the three scheme columns and combine their mappings
    for scheme_col in schemes.index:
        scheme_string = schemes[scheme_col]
        
        # Check if the string is valid (not NaN or empty)
        if pd.notna(scheme_string) and scheme_string.strip():
            
            # Parse the string into a dictionary
            current_mapping = parse_scheme_string(scheme_string=scheme_string)
            
            # Update the combined dictionary. 
            # Note: This uses the standard Python dict update/merge behavior. 
            # If a code appears in both Value Scheme and Missing Scheme, the mapping from the 
            # last processed scheme column will overwrite previous ones.
            combined_dict.update(current_mapping)

    # 3. Final check and return
    if not combined_dict:
        print(f"⚠️ Warning: Variable '{column_name}' found, but all scheme columns were empty.")
        return None
        
    return combined_dict


def replace_column_values(column_data: pd.Series, col_dict: dict) -> pd.Series:
    """
    Replaces numerical/coded values in a column (Series) with descriptive labels 
    from a mapping dictionary and converts any remaining NaNs to the string 'Missing'.

    Args:
        column_data (pd.Series): The column data (Series) to be cleaned.
        col_dict (dict): The mapping dictionary {code: label}.

    Returns:
        pd.Series: The column data with values replaced and NaNs converted to 'Missing'.
    """

    # 1. Prepare the mapping dictionary to handle float values
    # We must include float versions of integer/float keys for reliable matching.
    combined_map = col_dict.copy()
    float_map = {
        float(k): v 
        for k, v in col_dict.items() 
        if isinstance(k, (int, float))
    }
    combined_map.update(float_map)
    
    # 2. Perform the replacement of codes with labels
    # NaN values are preserved during this step.
    replaced_column = column_data.replace(combined_map)
    
    # 3. Replace remaining NaN values with the string 'Missing'
    final_column = replaced_column.fillna('Missing')
    
    return final_column