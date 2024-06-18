import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pickle

regions = ['MIR', 'NIR']  # 'MIR' or 'NIR'

dataset = 'Other_PET_PLA_PBAT_PHB'

folder_path = f'C:/Users/XXXX/{dataset}'
NIR_reference_reflectance_file_path = os.path.join(folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_shuffled_NIR.csv')
NIR_reference_reflectance = pd.read_csv(NIR_reference_reflectance_file_path)
MIR_reference_reflectance_file_path = os.path.join(folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_shuffled_MIR.csv')
MIR_reference_reflectance = pd.read_csv(MIR_reference_reflectance_file_path)

# looks for sample names that end with non-numeric number (eg .._asis.Sample.Raw, ...tched.Sample.Raw) and removes the last 16 characters
NIR_reference_reflectance['SampleName'] = NIR_reference_reflectance['SampleName'].apply(lambda x: x[:-11] if not x[-1:].isdigit() else x)


stand_devs = np.concatenate((np.arange(0,20,2), np.arange(20,50,10)), axis = 0)


for s in stand_devs:
    s =s/10

    for IR in regions:

        if IR == 'MIR':
            reference_reflectance = MIR_reference_reflectance.copy()
        if IR == 'NIR':
            reference_reflectance = NIR_reference_reflectance.copy()

        shuffled_order = reference_reflectance['SampleName']
        # Open relevant files
        residue_folder_path = f'C:/Users/XXXXXX/Food Noised Datasets/{dataset}'
        reflectance_file_path = os.path.join(residue_folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_{IR}_STD_{s}.csv')
        reflectance = pd.read_csv(reflectance_file_path)
        labels_file_path = os.path.join(residue_folder_path, f'PolyesterTypeLabels_{IR}_STD_{s}.csv')
        labels = pd.read_csv(labels_file_path)

        # removes '_STD_x.x'
        reflectance['SampleName'] = reflectance['SampleName'].apply(lambda x: x[:-8])
        labels['SampleName'] = labels['SampleName'].apply(lambda x: x[:-8])

        reflectance['SampleName'] = pd.Categorical(reflectance['SampleName'], categories=shuffled_order, ordered=True)
        reflectance_shuffled = reflectance.sort_values('SampleName')
        reflectance_shuffled.reset_index(drop=True, inplace=True)

        labels['SampleName'] = pd.Categorical(labels['SampleName'], categories=shuffled_order, ordered=True)
        labels_shuffled = labels.sort_values('SampleName')
        labels_shuffled.reset_index(drop=True, inplace=True)

        # Save shuffled reflectance and label values
        reflectance_shuffled.to_csv(f'{folder_path}/CombinedSpectraDataFrame_MaxMinNorm_shuffled_{IR}_STD_{s}.csv', index=False)
        labels_shuffled.to_csv(f'{folder_path}/PolyesterTypeLabels_shuffled_{IR}_STD_{s}.csv', index=False)



