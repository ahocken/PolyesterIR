import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

IR = 'MIR' # 'MIR' or 'NIR'

# specify what material you are trying to generate synthetic data for
material = 'PBAT'

# specify the number of total spectra desired (including real spectra)
num_desired_spectra = 170 

# Specify the folder path containing the CSV files
folder_path = f'C:/Users/XXXX/{IR} Data'

# Initialize an empty dataframe
combined_df = pd.DataFrame()

# Flag to keep track of the first file
is_first_file = True

# Specify material directory path to folder that holds real spectra 
material_folder_path = os.path.join(folder_path, material)

# MIR spectral files have no header, while NIR files do
if IR =='NIR':
     rows_to_skip = 1
     include_header = True
     ir_characters = int(-20)
if IR =='MIR':
     rows_to_skip = 0
     include_header = False
     ir_characters = -9

# This section looks at all the real spectra and finds the std for each wavelength
for file_name in os.listdir(material_folder_path):
    if file_name.endswith('.CSV') or file_name.endswith('.csv'):
        # Specify file path within materials directory
        file_path = os.path.join(material_folder_path, file_name)
    
        # Load data from real spectra file and store in dataframe
        data = np.loadtxt(file_path, delimiter=',', skiprows=rows_to_skip)
        df = pd.DataFrame(data)
        
        # Extract the column header from the first file
        if is_first_file:
                wavelengths = df.iloc[:, 0].tolist()
                is_first_file = False
        
        # Extract the second column (which holds the %R signals)
        second_column = df.iloc[:, 1]

        # Add the second column to the combined dataframe
        combined_df[file_name] = second_column

# Transpose the combined dataframe
combined_df = combined_df.transpose()

# Make columns headers the wavelengths
combined_df.set_axis(wavelengths, axis=1)

# combined_df = combined_df.drop('PBAT_2_asis.Sample.Raw.csv')

std_df = pd.DataFrame(columns = ['nm', 'std'])
std_df['nm'] = wavelengths
std_df['std'] = combined_df.std()

# find number of synthetic spectra needed
num_syn_spectra = num_desired_spectra - len(combined_df)

# this next section is where the synthetic data is created
for real_sample_file_name in os.listdir(material_folder_path):
    print(real_sample_file_name)

    # specify real sample file path within material directory
    real_sample_file_path = os.path.join(material_folder_path, str(real_sample_file_name))

    # Load data and store it in a dataframe
    data = np.loadtxt(real_sample_file_path, delimiter=',', skiprows=rows_to_skip)
    df = pd.DataFrame(data, columns = ['nm', '%R'])
    # add column to dataframe that holds the std for each wavelength
    df['std'] = std_df['std']

    

    x = df['nm']
    y = df['%R']

    # number of synthetic spectra created based on each real spectra
    num_variations = int(num_syn_spectra/len(combined_df))
    counter = 1
    
    for _ in range(num_variations):

        # initialize empty lists to hold synthetic data
        synthetic_nm = []
        synthetic_r = []

        # Iterate over each wavelength
        for index, row in df.iterrows():

            # sample_std = min(row['std'], 1)
            sample_std = row['std']
            # Generate synthetic %R value by sampling from a normal distribution (loc = mean, scale = std)
            synthetic_r_value = np.random.normal(loc=row['%R'], scale=sample_std)
            # Append wavelength and %R to lists
            synthetic_nm.append(row['nm'])
            synthetic_r.append(synthetic_r_value)

        
        # Create synthetic spectra DataFrame
        synthetic_data = pd.DataFrame({'nm': synthetic_nm, '%R': synthetic_r})


        # save sythetic spectra in a csv file 
        synthetic_data.to_csv(f'{material_folder_path[:-4]}/Syn{material}/{real_sample_file_name[:ir_characters]}_normal_std_{counter}.csv', index=False, header=include_header)
        counter = 1 + counter      

