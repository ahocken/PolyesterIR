import pandas as pd
import numpy as np
import os
import random


regions = ['MIR','NIR']  
dataset = 'PET_PLA_PBAT_PHB'

folder_path = f'C:/Users/XXXXX/{dataset}'
reflectance_file_path = os.path.join(folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_NIR.csv')
reflectance = pd.read_csv(reflectance_file_path)

shuffled_indices = reflectance.index.to_list()
random.shuffle(shuffled_indices)

for IR in regions:
    # Open relevant files
    folder_path = f'C:/Users/XXXXXX/{dataset}'
    reflectance_file_path = os.path.join(folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_{IR}.csv')
    reflectance = pd.read_csv(reflectance_file_path)
    labels_file_path = os.path.join(folder_path, f'PolyesterTypeLabels_{IR}.csv')
    labels = pd.read_csv(labels_file_path)

    # Shuffle and reset index numbers the same way using shuffled indices
    reflectance_shuffled = reflectance.iloc[shuffled_indices].reset_index(drop=True)
    labels_shuffled = labels.iloc[shuffled_indices].reset_index(drop=True)

    # Save shuffled reflectance and label values
    reflectance_shuffled.to_csv(f'{folder_path}/CombinedSpectraDataFrame_MaxMinNorm_shuffled_{IR}.csv', index=False)
    labels_shuffled.to_csv(f'{folder_path}/PolyesterTypeLabels_shuffled_{IR}.csv', index=False)


# Identify unique classes in y
y = labels_shuffled.drop('SampleName', axis=1)
classes = np.unique(y)

n_splits = 6
# Initialize lists to store indices for each subset
subset_indices = [[] for _ in range(n_splits)]

# Stratified sampling
for class_label in classes:
    # Find indices of samples with this class label
    indices = np.where(y == class_label)[0]
    # Split the indices into 6 equal parts
    split_indices = np.array_split(indices, n_splits)
    # Append to corresponding subset indices
    for i, split_index in enumerate(split_indices):
        subset_indices[i].extend(split_index)

# Convert subset indices to arrays
subset_indices = [np.array(indices) for indices in subset_indices]
subset_indices_structured = np.array(subset_indices, dtype=object)
np.save(f'{folder_path}/split_indicies.npy', subset_indices_structured, allow_pickle=True)
