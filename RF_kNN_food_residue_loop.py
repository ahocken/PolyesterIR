import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
import pickle
import torch


def Average(lst): 
    return sum(lst) / len(lst) 


# Specify IR region
IR = 'NIR'  # 'MIR' or 'NIR'
dataset = 'Other_PET_PLA_PBAT_PHB'

folder_path = f'C:/Users/XXXXX/{dataset}'
result_folder_path = os.path.join(folder_path, f'RF_{IR}_results', f'redos')

subset_indices = np.load(f'{folder_path}/split_indicies.npy', allow_pickle=True)

stand_devs = np.concatenate((np.arange(0,20,2), np.arange(20,50,10)), axis = 0)
rf_score_means = []
rf_score_stds = []
knn_score_means = []
knn_score_stds =[]

for s in stand_devs:
    s = s/10
    print(f'Beginning: {s}')

    residue_reflectance_file_path = os.path.join(folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_shuffled_{IR}_STD_{s}.csv')
    residue_reflectance = pd.read_csv(residue_reflectance_file_path)
    labels_file_path = os.path.join(folder_path, f'PolyesterTypeLabels_shuffled_{IR}_STD_{s}.csv')
    labels = pd.read_csv(labels_file_path)

    y = labels.drop('SampleName', axis =1)
    X = residue_reflectance.drop('SampleName', axis=1)

    test_accuracies_RF = []
    test_accuracies_kNN = []

    for i in range(len(subset_indices)):

        subset_indices = np.load(f'{folder_path}/split_indicies.npy', allow_pickle=True)
        test_indices = subset_indices[i]

        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        # Convert data to PyTorch tensors
        X_test_tensor = torch.tensor(X_test.values)
        y_test_tensor = torch.tensor(y_test.values.ravel(), dtype=torch.long)

        # Convert tensors to numpy arrays
        X_test_numpy = X_test_tensor.numpy()
        y_test_numpy = y_test_tensor.numpy()

        # Load models for this split
        with open(f'{folder_path}/Best Models/best_model_RF_{IR}_Split{i+1}.pkl', 'rb') as f:
            best_model_RF = pickle.load(f)

        with open(f'{folder_path}/Best Models/best_model_kNN_{IR}_Split{i+1}.pkl', 'rb') as f:
            best_model_kNN = pickle.load(f)

        test_predictions_split_RF = best_model_RF.predict(X_test_numpy)
        test_accuracy_split_RF = accuracy_score(y_test_numpy, test_predictions_split_RF)
        test_accuracies_RF.append(test_accuracy_split_RF)

        test_predictions_split_kNN = best_model_kNN.predict(X_test_numpy)
        test_accuracy_split_kNN = accuracy_score(y_test_numpy, test_predictions_split_kNN)
        test_accuracies_kNN.append(test_accuracy_split_kNN)

    knn_score_means.append(np.mean(test_accuracies_kNN))
    knn_score_stds.append(np.std(test_accuracies_kNN))
    rf_score_means.append(np.mean(test_accuracies_RF))
    rf_score_stds.append(np.std(test_accuracies_RF))

fig, ax1 = plt.subplots(1,1, figsize=(5,5))
ax1.errorbar(stand_devs/10, rf_score_means, yerr=rf_score_stds, fmt='o', color='#388968', capsize=2, label='Random Forest')
ax1.errorbar(stand_devs/10, knn_score_means, yerr=knn_score_stds, fmt='s', color='#FADE83', capsize=2, label='kNN')
ax1.set_xlabel('Guassian Noise Standard Deviation', fontsize = 12)
ax1.set_ylabel('Prediction Accuracy', fontsize=12)
ax1.tick_params(axis = 'both', which='major', labelsize=12)
ax1.legend(frameon=False, fontsize= 12)

fig.savefig(f'{result_folder_path}/food_residue_{IR}_knn_RF_fig.pdf')