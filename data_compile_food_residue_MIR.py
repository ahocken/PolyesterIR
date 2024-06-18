import pandas as pd
import os
import numpy as np

main_folder_path = r'C:/Users/XXXXXX/Food Noised Samples/MIR'

# These material names should be the same as their folder name in your directory
materials = ['Other','PET', 'PLA', 'PBAT', 'SynPBAT', 'PHB', 'SynPHB']
stand_devs = np.concatenate((np.arange(0,20,2), np.arange(20,50,10)), axis = 0)

for i in stand_devs:
    i = i/10
    print(f'Beginning: {i}')
    std_folder_path = main_folder_path + f'/Food Noised STD{i}'

    combined_df = pd.DataFrame()

    # Flag to keep track of the first file
    is_first_file = True

    # Initialize an empty array for sample names
    sample_names = []

    # Initialize an empty array of labels (trash vs polyester)
    labels = []
    # Initialize an empty array for labels (trash vs PET vs PLA vs PBAT)
    labels_polyester = []

    for material in materials:
        material_folder_path = os.path.join(std_folder_path, material)
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
                    column_headers = df.iloc[415:3941, 0].tolist()
                    is_first_file = False
                    
                # Extract the second column
                second_column = df.iloc[415:3941, 1]
                max = np.max(second_column)
                min = np.min(second_column)

                second_column = (second_column - min)/(max - min)

                # Add the second column to the combined dataframe
                combined_df[file_name] = second_column

                if file_name.startswith('Other'):
                    labels_polyester.append(0)
                    labels.append(0)
                if file_name.startswith('PET'):
                    labels_polyester.append(1)
                    labels.append(1)
                if file_name.startswith('PLA'):
                    labels_polyester.append(2)
                    labels.append(1)
                if file_name.startswith('PBAT'):
                    labels_polyester.append(3)
                    labels.append(1)
                if file_name.startswith('PHB'):
                    labels_polyester.append(4)
                    labels.append(1)

    # Transpose the combined dataframe
    combined_df = combined_df.transpose()

    # Assign the column headers to the combined dataframe
    combined_df.columns = column_headers

    # Add a column to beginning of dataframe for sample names
    combined_df.insert(loc=0, column = 'SampleName', value = sample_names)

    labels_polyester.insert(0, 'PolyesterTypeLabel')
    labels.insert(0,'TrashvsPolyester')
    # Export the combined dataframe to a CSV file
    # result_folder_path = f'A:/Sharona Huang/UROP/Result Folders/Combinations/Other, PET, PLA, PBAT_Noised_STD_0.0/Food Residue Noised Versions/Residue_Noised_STD_{i}'
    output_path = os.path.join(main_folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_MIR_STD_{i}.csv')
    combined_df.to_csv(output_path, index=False)

    sample_names.insert(0,'SampleName')

    output_path_labels = os.path.join(main_folder_path, f'TrashVPolyesterLabels_MIR_STD_{i}.csv')
    np.savetxt(output_path_labels, #file path
            np.transpose([sample_names,labels]), # content/data to be added to csv file
            delimiter =", ", # set delimiter to be a comma followed by a space
            fmt ='% s')  # converts format of data to a str
    output_path_labels_polyester = os.path.join(main_folder_path, f'PolyesterTypeLabels_MIR_STD_{i}.csv')
    np.savetxt(output_path_labels_polyester, #file path
            np.transpose([sample_names,labels_polyester]), # content/data to be added to csv file
            delimiter =", ", # set delimiter to be a comma followed by a space
           fmt ='% s')  # converts format of data to a str
