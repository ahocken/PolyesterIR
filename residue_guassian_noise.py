
import os
import numpy as np
import pandas as pd
import random

IR = 'NIR' #'NIR' or 'MIR"

if IR == 'NIR':
    inlcude_header = True
else:
    include_header = False

# Specify the folder path containing the CSV files
folder_path = f'C:/Users/XXXXX'
materials = ['Other','PET', 'PLA', 'PBAT', 'SynPBAT', 'PHB', 'SynPHB']

stand_devs = np.concatenate((np.arange(0,20,2), np.arange(20,50,10)), axis = 0)

#iterates through upper threshold for gaussian
for i in stand_devs:
    i = i/10
    i = float(i)
    print(f'Beginning STD: {i}')

    # make folder for this STD if it doesn't already exist
    std_folder_path = folder_path + f'Food Noised Samples/{IR}/Food Noised STD{i}/'
    if not os.path.exists(std_folder_path):
        os.makedirs(std_folder_path)

    for material in materials:
        result_material_folder_path = os.path.join(std_folder_path, material)
        if not os.path.exists(result_material_folder_path):
            os.makedirs(result_material_folder_path)

        print(f'Beginning: {material}')
        material_folder_path = os.path.join(folder_path, f'{IR} Data', material)        

        for file_name in os.listdir(material_folder_path):
            if file_name.endswith('.CSV') or file_name.endswith('.csv'):
                file_path = os.path.join(material_folder_path, str(file_name))
                data = np.loadtxt(file_path, delimiter=',', skiprows=1)
                df = pd.DataFrame(data)

                x = df[0]
                y = df[1]
                
                std = round(random.uniform(0, i), 3)

                # Generate noise with same size as that of the data.
                guas_noise = np.random.normal(0,std, len(x)) #  μ = 0, σ = 0.1, size = length of x or y

                # Add the noise to the data.
                y_noised = y + guas_noise

                df[1] = y_noised

                df.columns = ['nm', '%R']

                i_str = str(i).replace('.','')
                
                if file_name[-5:-4].isdigit() or IR == 'MIR':
                    result_file_path = result_material_folder_path + '/' + file_name[:-4] + f'_STD_{i}.csv'
                else:
                    result_file_path = result_material_folder_path + '/' + file_name[:-15] + f'_STD_{i}.csv'
                df.to_csv(result_file_path, index=False, header=include_header)




