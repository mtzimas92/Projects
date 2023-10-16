import re
import pandas as pd
import numpy as np
def extract_deformation_data(data_directory, file_name, node_id):
    """
    Extracts deformation data for a specified node from a given file.
    
    Args:
        data_directory (str): Path to the directory containing the file.
        file_name (str): Name of the file containing deformation data.
        node_id (str): The specified node ID for which deformation data is needed.
        
    Returns:
        pd.DataFrame: DataFrame containing deformation data.
    """


    # Define an empty list for deformation data
    deformation_list = []

    # Construct the full path to the file
    file_path = f'{data_directory}/{file_name}'

    # Read the file and extract deformation information
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('0'):
                # Move to the table with the deformation information
                for _ in range(4):
                    next(f)
                # Read the table row by row and extract the deformation information for the specified node
                for line in f:
                    if line.startswith("          " + node_id):  
                        deformation = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', line)
                        deformation_list.append(list(map(float, deformation[1:])))

    # Create a DataFrame from the deformation data
    deformation_dataframe = pd.DataFrame(deformation_list)
    deformation_dataframe.columns =['T1', 'T2', 'T3', 'R1', 'R2','R3']

    return deformation_dataframe

# Example Usage
data_directory = 'C:/NASTRAN_TEST/FLUTTER'
file_name = 'flutter_summary.f06'
node_id = '<node_id>'  # Replace with the actual node ID you are interested in
result_dataframe = extract_deformation_data(data_directory, file_name, node_id)


def extract_flutter_summary(file_path):
	"""
    Extracts flutter summary information from a specified file.
    
    Args:
        file_path (str): Path to the file containing flutter summary information.
        
    Returns:
        list, list, list: Lists of DataFrames, column names, and ratios.
    """
    with open(file_path, 'r') as f:
        # Create empty lists to hold the column names and dataframes
        column_names_list = []
        dfs = []
        ratios = []
        
        # Flag to indicate whether we're currently in a flutter summary section
        in_flutter_summary = False
        
        # Iterate over each line in the file
        for line in f:
            if 'FLUTTER  SUMMARY' in line:
                in_flutter_summary = True
                # Skip the next few lines, which are just separators and units
                next(f) # separator
                ratios.append(float((next(f).split()[10])))
                next(f) # separator
                next(f) # separator
                # The column names are in the next line
                column_names = next(f).split()
                column_names_list.append(column_names)
                # Create a new empty dataframe for this flutter summary section
                df = pd.DataFrame(columns=column_names)
                dfs.append(df)
                
            elif in_flutter_summary:
                if 'NASTRAN' in line:
                    in_flutter_summary = False
                elif 'FORTRAN' in line or 'WORDS' in line:
                    continue
                else:
                    # Split the line into columns and append to the data list
                    row = line.split()
                    if len(row) == len(df.columns):
                        df.loc[len(df)] = row
                    
    return dfs, column_names_list, ratios

# Define the file path
file_path = 'path_to_your_file/flutter_summary.f06'  # Replace with the actual file path

# Call the function and store the results
dfs, column_names_list, ratios = extract_flutter_summary(file_path)

#transform to numeric from strings
for i in range(len(dfs)):
    dfs[i] = dfs[i].apply(pd.to_numeric)
	
def process_material_data(material_type, young, limit_stress, rhorat, result_dataframe, dfs, num_eigenvalues):
    """
    Processes material data for a specified material type and generates features and labels for machine learning.

    Args:
        material_type (str): Type of material (e.g., '7075', '6061', '2024').
        young (float): Young's Modulus for the material.
        limit_stress (float): Limiting Stress for the material.
        rhorat (list): List of values related to the material.
        result_dataframe (pd.DataFrame): DataFrame containing deformation data.
        dfs (list): List of DataFrames for the material.
        num_eigenvalues (int): Number of eigenvalues requested.

    Returns:
        list, list, list: Lists of features, binary labels, and labels for machine learning.

    The function processes material data for a specified material type, generating features and labels suitable for
    machine learning tasks. It iterates through DataFrames to extract relevant information, combines it with material
    properties, and constructs feature vectors. Binary labels are determined based on damping values, and labels are
    derived from the damping values.
    """
    features = []
    binary_labels = [] 
    labels = [] 
    
    for j, f in enumerate(dfs):
        for k in range(len(f.iloc[:]['VELOCITY'])):
            for eigen_index in range(num_eigenvalues):
                if j < num_eigenvalues:
                    features.append([young, limit_stress, chord, span, area, rhorat[j],
                                    result_dataframe.iloc[j]['T1'], result_dataframe.iloc[j]['T2'],
                                    result_dataframe.iloc[j]['T3'], result_dataframe.iloc[j]['R1'],
                                    result_dataframe.iloc[j]['R2'], result_dataframe.iloc[j]['R3'],
                                    f.iloc[k]['VELOCITY'], f.iloc[k]['1./KFREQ'], f.iloc[k]['EIGENVALUE']])
                    
                    if f.iloc[k]['DAMPING'] < 0:
                        binary_labels.append(0)
                    else:
                        binary_labels.append(1)
                    labels.append(f.iloc[k]['DAMPING'])
                
                else:
                    eigen_index_mod = eigen_index % num_eigenvalues
                    features.append([young, limit_stress, chord, span, area, rhorat[j],
                                    result_dataframe.iloc[eigen_index_mod]['T1'], result_dataframe.iloc[eigen_index_mod]['T2'],
                                    result_dataframe.iloc[eigen_index_mod]['T3'], result_dataframe.iloc[eigen_index_mod]['R1'],
                                    result_dataframe.iloc[eigen_index_mod]['R2'], result_dataframe.iloc[eigen_index_mod]['R3'],
                                    f.iloc[k]['VELOCITY'], f.iloc[k]['1./KFREQ'], f.iloc[k]['EIGENVALUE']])
                    
                    if f.iloc[k]['DAMPING'] < 0:
                        binary_labels.append(0)
                    else:
                        binary_labels.append(1)
                    labels.append(f.iloc[k]['DAMPING'])
    
    return features, binary_labels, labels

# Example usage for 7075 material data with 6 eigenvalues
num_eigenvalues = 6
features7075, binary_labels7075, labels7075 = process_material_data(
    "7075", young_7075, limit_stress_7075, rhorat1, result_dataframe, dfs_7075, num_eigenvalues)
	
np.savetxt('features7075.txt',features7075)
np.savetxt('labels7075.txt',labels7075)
np.savetxt('binary_labels7075.txt',binary_labels7075)
