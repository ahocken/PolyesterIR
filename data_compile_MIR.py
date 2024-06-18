## This code imports a folder of spectra csv files and extracts the reflectance
## data (second column of csv) and compiles it into a single data frame which 
## is then exported to a CSV file called CombinedSpeactraDataFrame. Based on
## the first three letters of the file name, the code assigns label (1, 2, or 3) 
## and exports the list of labels to a CSV file called SpacetraLabels. 

## CSV file inputs should have two columns (wavelength and reflectance, respectively)
## with headers. This is how the MRL UV/NIR exports its data to CSV

import pandas as pd
import os
import numpy as np


# Specify the folder path containing the CSV files
folder_path = r'C:/Users/XXXXXX/MIR Data'

# These material names should be the same as their folder name in your directory
materials = ['PET', 'PLA', 'PBAT', 'SynPBAT', 'PHB', 'SynPHB']
dataset = 'Other_PET_PLA_PBAT_PHB'

# Initialize an empty dataframe
combined_df = pd.DataFrame()

# Flag to keep track of the first file
is_first_file = True

# Initialize an empty array for sample names
sample_names = []

# Initialize an empty array for labels (trash vs PET vs PLA vs PBAT)
labels_polyester = []


for material in materials:
    material_folder_path = os.path.join(folder_path, material)
    print(f'Beginning to add: {material}')

    # Iterate through each file in the folder
    for file_name in os.listdir(material_folder_path):
        if file_name.endswith('.CSV') or file_name.endswith('.csv'):
            file_path = os.path.join(material_folder_path, file_name)
            sample_names.append(file_name[0:-4])
    
        
            data = np.loadtxt(file_path, delimiter=',')
            df = pd.DataFrame(data)
        
            # Extract the column header from the first file
            if is_first_file:
                column_headers = df.iloc[416:3942, 0].tolist()
                is_first_file = False
        
            # Extract the second column
            second_column = df.iloc[416:3942, 1]
            maxi = np.max(second_column)
            mini = np.min(second_column)

            second_column = (second_column - mini)/(maxi - mini)
        
            # Add the second column to the combined dataframe
            combined_df[file_name] = second_column

            if file_name.startswith('Other'): 
                labels_polyester.append(0)    
            if file_name.startswith('PET'):
                labels_polyester.append(1)
            if file_name.startswith('PLA'):
                labels_polyester.append(2)
            if file_name.startswith('PBAT'):
                labels_polyester.append(3)
            if file_name.startswith('PHB'):
                labels_polyester.append(4)


# Transpose the combined dataframe
combined_df = combined_df.transpose()

# Assign the column headers to the combined dataframe
combined_df.columns = column_headers

# Add a column to beginning of dataframe for sample names
combined_df.insert(loc=0, column = 'SampleName', value = sample_names)

labels_polyester.insert(0, 'PolyesterTypeLabel')

# Export the combined dataframe to a CSV file
result_folder_path = 'C:/Users/XXXXX/{dataset}'
output_path = os.path.join(result_folder_path, 'CombinedSpectraDataFrame_MaxMinNorm_MIR.csv')
combined_df.to_csv(output_path, index=False)

sample_names.insert(0,'SampleName')

output_path_labels_polyester = os.path.join(result_folder_path, 'PolyesterTypeLabels_MIR.csv')
np.savetxt(output_path_labels_polyester, #file path
           np.transpose([sample_names,labels_polyester]), # content/data to be added to csv file
           delimiter =", ", # set delimiter to be a comma followed by a space
           fmt ='% s')  # converts format of data to a str