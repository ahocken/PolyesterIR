import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle

IR = 'MIR' # 'NIR' or 'MIR'
dataset = 'Other_PET_PLA_PBAT_PHB'  # 'Other_PET_PLA_PBAT_PHB' or 'Other_PET_PLA'

folder_path = f'C:/Users/XXXXX/{dataset}'
reflectance_file_path = os.path.join(folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_shuffled_{IR}.csv')
reflectance = pd.read_csv(reflectance_file_path)
labels_file_path = os.path.join(folder_path, f'PolyesterTypeLabels_shuffled_{IR}.csv')
labels = pd.read_csv(labels_file_path)

feature_importance_file_path = os.path.join(folder_path, f'RF_{IR}_results', f'RF_feature_importances_{IR}.csv')
feature_importance = pd.read_csv(feature_importance_file_path)

y = labels.drop('SampleName', axis = 1)
X = reflectance.drop('SampleName', axis=1)

subset_indices = np.load(f'{folder_path}/split_indicies.npy', allow_pickle=True)

if IR == 'NIR':
    thresholds = np.concatenate((np.arange(0, 0.001, 0.0001),np.arange(0.001,0.037,0.001)))
if IR == 'MIR':
    thresholds = np.concatenate((np.arange(0, 0.001, 0.0001),np.arange(0.001,0.037,0.001)))

accuracy_scores_avg_list = []
accuracy_scores_std_list = []
n_features = []
reduced_feature_names = []


for t in range(len(thresholds)):

    test_accuracies_thresh = []
    for i in range(len(subset_indices)):
        subset_indices = np.load(f'{folder_path}/split_indicies.npy', allow_pickle=True)

        # importance features
        nonimportant_features_boolean = (feature_importance.loc[[i]] <= thresholds[t])
        nonimportant_features = [s for s,keep in zip(X.columns.to_list(), nonimportant_features_boolean.iloc[0].values) if keep]

        X_reduced = X.drop(columns = nonimportant_features)

        if X_reduced.empty:
            break


        
        test_indices = subset_indices[i]
        train_indices_subsets = np.delete(subset_indices,i)
        train_indices = np.concatenate(train_indices_subsets, axis=None)

        # Manual test_train_split
        X_train = X_reduced.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X_reduced.iloc[test_indices]
        y_test = y.iloc[test_indices]

        with open(f'{folder_path}/Best Models/best_model_RF_{IR}_Split{i+1}.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        best_params = best_model.get_params()
        # define classificaitons
        rf = RandomForestClassifier(**best_params)
        rf.fit(X_train,y_train)
        test_accuracy_split = rf.score(X_test,y_test)
        test_accuracies_thresh.append(test_accuracy_split)

    accuracy_scores_avg_list.append(np.mean(test_accuracies_thresh))
    accuracy_scores_std_list.append(np.std(test_accuracies_thresh))
    reduced_feature_names.append(X_reduced.columns)
    n_features.append(len(X_reduced.columns))
    print(f'Number of features: {len(X_reduced.columns)}')

plt.figure(figsize=(5,5))
plt.scatter(n_features,[x * 100 for x in accuracy_scores_avg_list], color = '#388968')
plt.errorbar(n_features,[x * 100 for x in accuracy_scores_avg_list], yerr=[x* 100 for x in accuracy_scores_std_list], capsize=2, fmt="o", color = '#388968')
plt.xlabel('Number of Features', fontsize =12)
plt.ylabel('Prediction Accuracy (%)', fontsize = 12)
plt.ylim([80,100])
plt.yticks(np.arange(80, 101, 5), fontsize = 12)
plt.tick_params(axis='both', labelsize=12,direction='in')


if IR == 'NIR':
    plt.xticks(np.arange(0, max(n_features)+1, 50))
if IR == 'MIR':
    plt.xticks(np.arange(0, max(n_features)+100, 500))


plt.savefig(os.path.join(folder_path, f'RF_{IR}_results', f'RF_reduced_feature_{IR}.pdf'))

plt.show()