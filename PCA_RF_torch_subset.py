import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

########################################################################################################################
# Split data based on a given list of indices
class CustomSplitter(BaseCrossValidator):
    def __init__(self, indices_list):
        self.indices_list = indices_list

    def _iter_test_indices(self, X=None, y=None, groups=None):
        for test_indices in self.indices_list:
            yield test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.indices_list)
    
###################################################################################################################

# Specify IR region
IR = 'NIR'  # 'MIR' or 'NIR'

class_names = ['Other','PET', 'PLA', 'PBAT', 'PHB']
dataset = 'Other_PET_PLA_PBAT_PHB'

if class_names[0] == 'Other':
    class_names_wo_other = class_names[1:]
else:
    class_names_wo_other = class_names

if class_names[-1] == 'PHB':
    if IR == 'MIR':
        n_components = 8
    elif IR == 'NIR':
        n_components = 5
if class_names[-1] == 'PLA':
    n_components = 4


folder_path = f'C:/Users/XXXXX/{dataset}'
result_folder_path = os.path.join(folder_path, f'PCA_RF_{IR}_results')
reflectance_file_path = os.path.join(folder_path, f'CombinedSpectraDataFrame_MaxMinNorm_shuffled_{IR}.csv')
reflectance = pd.read_csv(reflectance_file_path)
labels_file_path = os.path.join(folder_path, f'PolyesterTypeLabels_shuffled_{IR}.csv')
labels = pd.read_csv(labels_file_path)

y = labels.drop('SampleName', axis=1)
X = reflectance.drop('SampleName', axis=1)

subset_indices = np.load(f'{folder_path}/split_indicies.npy', allow_pickle=True)

# Define the parameter grid
param_grid = {
    'n_estimators': [int(2**x) for x in np.arange(0,11,1)],
    'max_depth': [1,2,4,8,16,32,64,128,256],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_leaf_nodes' : [2,4,8,16,32,64,128,256,512]
}


# Initialization of variables to store values later
best_params = []  # Stores the best parameters found in grid_search for each split
confusion_matrices = []  # Stores the confusion matrices for each test split (should have 6 conf. mats.)
test_accuracies = []
thresh_accuracies = []
percentage_samples_lost = []

# Create a dataframe that holds the sample names for each index, their true class label, and their predicted label from the test splits
prediction_df = pd.DataFrame(columns=['SampleName', 'True Label', 'Predicted Label'])
prediction_df['SampleName'] = reflectance['SampleName']
prediction_df['True Label'] = labels[' PolyesterTypeLabel']

# Initialize a DataFrame with feature names to store their importance scores for each model
initial_feature_names = X.columns
initial_feature_names = [eval(i) for i in initial_feature_names]
feature_importance_df = pd.DataFrame(columns = initial_feature_names)

for i in range(len(subset_indices)):
    subset_indices = np.load(f'{folder_path}/split_indicies.npy', allow_pickle=True)
    test_indices = subset_indices[i]
    train_indices_subsets = np.delete(subset_indices,i)
    train_indices = np.concatenate(train_indices_subsets, axis=None)

    # Manual test_train_split
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values)
    y_train_tensor = torch.tensor(y_train.values.ravel(), dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values)
    y_test_tensor = torch.tensor(y_test.values.ravel(), dtype=torch.long)

    # Convert tensors to numpy arrays
    X_train_numpy = X_train_tensor.numpy()
    y_train_numpy = y_train_tensor.numpy()
    X_test_numpy = X_test_tensor.numpy()
    y_test_numpy = y_test_tensor.numpy()

    pca = PCA(n_components=n_components)
    pca.fit(X_train_numpy)

    # Save PCA model for this split
    with open(f'{folder_path}/PCA Models/PCA_RF_{IR}_Split{i+1}.pkl', 'wb') as f:
        pickle.dump(pca,f)

    X_train_numpy_pca = pca.transform(X_train_numpy)
    X_test_numpy_pca = pca.transform(X_test_numpy)

    # Specify classification model
    rf = RandomForestClassifier()

    # Create dictionary to map original indices to positional indices
    index_dict = {}
    index_dict = dict(list(enumerate(train_indices)))
    reverse_dict = {value: key for key, value in index_dict.items()}
    train_indices_subsets_updated = train_indices_subsets.copy()
    for r in range(len(train_indices_subsets)):
        for c in range(len(train_indices_subsets[r])):
            train_indices_subsets_updated[r][c] = reverse_dict[train_indices_subsets[r][c]]

    # Define how folds are split
    custom_splitter = CustomSplitter(train_indices_subsets_updated)

    # Conduct grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=custom_splitter, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_numpy_pca,y_train_numpy)

    # Store grid search results
    grid_results = grid_search.cv_results_

    # Store parameters with best accuracy score
    best_params.append(grid_search.best_params_)

    # Store parameter combinations and associated accuracies in a dataframe; export as csv
    params_df = pd.DataFrame()
    params_df['Accuracy'] = grid_results['mean_test_score']
    params_df['max_depth'] = [grid_results['params'][n]['max_depth'] for n in range(len(grid_results['params']))]
    params_df['n_estimators'] = [grid_results['params'][n]['n_estimators'] for n in range(len(grid_results['params']))]
    params_df['max_leaf_nodes'] = [grid_results['params'][n]['max_leaf_nodes'] for n in range(len(grid_results['params']))]
    params_df.to_csv(f'{folder_path}/Hyperparameter Optimization Results/PCA_RF_grid_search_results_{IR}_split{i+1}.csv')
    
    # Take best model from grid search and use it to predict test subset
    best_model = grid_search.best_estimator_
    test_predictions_split = best_model.predict(X_test_numpy_pca)
    test_accuracy_split = accuracy_score(y_test_numpy, test_predictions_split)
    test_accuracies.append(test_accuracy_split)
    print(f'Split {i+1} Test Accuracy: {test_accuracy_split:.4f}')

    # Save best model for this split
    with open(f'{folder_path}/Best Models/best_model_PCA_RF_{IR}_Split{i+1}.pkl', 'wb') as f:
        pickle.dump(best_model,f)

    # Store confusion matrix for plotting later
    cm = confusion_matrix(y_test_numpy, test_predictions_split)
    confusion_matrices.append(cm)

    # Store prediction labels for sample characteristic analysis later
    test_predictions_split_series = pd.Series(test_predictions_split, index=test_indices)
    prediction_df['Predicted Label'] = prediction_df['Predicted Label'].combine_first(test_predictions_split_series)

    # Store prediction probabilities for probability threshold analysis later
    pred_probabilities = best_model.predict_proba(X_test_numpy_pca)
    for sub_arr in pred_probabilities:
        max_val = max(sub_arr)
        for z in range (len(sub_arr)):
            if sub_arr[z] != max_val:
                sub_arr[z] = 0
    
    # Specify probability thresholds to run
    prob_threshold = np.concatenate((np.arange(0,0.96,.05),np.arange(0.96,1.01,0.01)))

    # Make a binary list of the true labels
    binary_true_labels = np.zeros_like(pred_probabilities)
    for j in range(len(test_indices)):
        binary_true_labels[j][y[' PolyesterTypeLabel'][test_indices[j]]]=1

    thresh_accuracies_split = []
    percentage_samples_lost_split = []

    for thresh in prob_threshold:
        # Changes any probabilities larger than thresh to be 1
        binary_predictions = (pred_probabilities > thresh).astype(int)
        classified_samples_mask = np.any(binary_predictions,axis=1)
        # 'filtered' means that probabilities below threshold have been removed
        filtered_true_labels = y.iloc[test_indices][classified_samples_mask]
        filtered_binary_true_labels = binary_true_labels[classified_samples_mask]
        filtered_binary_predictions = binary_predictions[classified_samples_mask]
        thresh_accuracies_split.append(accuracy_score(filtered_binary_true_labels,filtered_binary_predictions))

        percentage_samples_lost_split.append(100 - (np.count_nonzero(classified_samples_mask)/len(classified_samples_mask) * 100))

    thresh_accuracies.append(thresh_accuracies_split)
    percentage_samples_lost.append(percentage_samples_lost_split)

##############################################################################################################    
## Sample Characteristic Analysis

# Import excel file containing sample characteristics with different sheets for each class
sample_descrip_file_path = os.path.join(folder_path, 'PolyesterSampleList_May22024.xlsx')
df_all = pd.read_excel(sample_descrip_file_path, sheet_name = class_names_wo_other)
sample_descrip = pd.concat(df_all, ignore_index=True)

# Create and fill column holding sample names
if IR == 'MIR':
    ch = -5
if IR == 'NIR':
    ch = -16
# looks for sample names that end with non-numeric number (eg .._asis.Sample.Raw, ...tched.Sample.Raw) and removes the last 16 characters
prediction_df['SampleName'] = prediction_df['SampleName'].apply(lambda x: x[:ch] if not x[-1:].isdigit() else x)
# looks for sample names that end with non-numeric number (eg .._scra) and removes the last 5 characters
prediction_df['SampleName'] = prediction_df['SampleName'].apply(lambda x: x[:-5] if not x[-1:].isdigit() else x)


# add in sample descriptions entries in results dataframe if they have the same SampleName (ignores and non-overlapping samples)
results = pd.merge(prediction_df, sample_descrip, on='SampleName')
# Change data type to match 'True label' data type
results['Predicted Label'] = results['Predicted Label'].astype(np.int64)

## Calculates accuracy scores and groups them by color
accuracy_scores = results.groupby('Color').apply(lambda x: accuracy_score(x['True Label'], x['Predicted Label']))
# puts accuracy scores into dataframe
color_accuracy_df = pd.DataFrame(accuracy_scores, columns = ['Accuracy'])
# intitialize to store the number of occurences of each color
color_counts = []
# count number of occurences of each color
for i in color_accuracy_df.index:
    color_counts.append(results['Color'].value_counts()[i])
# store color counts color accuracy dataframe
color_accuracy_df['OccurenceNumber'] = color_counts
# sort values in descending order
color_accuracy_df = color_accuracy_df.sort_values(by="Accuracy", ascending=False)
print("Accuracy scores by color:")
print(color_accuracy_df)

## Caluclates accuracy scores and groups them by transparency
accuracy_scores = results.groupby('Transparency').apply(lambda x: accuracy_score(x['True Label'], x['Predicted Label']))
# puts accuracy scores into dataframe
transmission_accuracy_df = pd.DataFrame(accuracy_scores, columns = ['Accuracy'])
# initialize to store the number of occurences of each transparency
transmission_counts = []
# count number of occurences of each transparency
for i in transmission_accuracy_df.index:
    transmission_counts.append(results['Transparency'].value_counts()[i])
# store transparency counts in transparency accuracy dataframe
transmission_accuracy_df['OccurenceNumber'] = transmission_counts
# sort values in descending order
transmission_accuracy_df = transmission_accuracy_df.sort_values(by="Accuracy", ascending=False)
print("Accuracy scores by transmission:")
print(transmission_accuracy_df)

## Calculates accuracy scores and groups them by sample thickness
increment = 0.2
bins = list(np.arange(0,3.1,increment))
# Create a new column in the DataFrame to hold the bin labels
results['Thickness Bin'] = pd.cut(results['Thickness Avg (mm)'], bins)
# Group the data by thickness bins and calculate accuracy score for each group
accuracy_scores = results.groupby('Thickness Bin').apply(lambda x: accuracy_score(x['True Label'], x['Predicted Label']))
# puts accuracy scores into dataframe
thickness_accuracy_df = pd.DataFrame(accuracy_scores, columns = ['Accuracy'])
# initialize to store the number of occurences of each thickness bin
thickness_counts, bin_edges= np.histogram(results['Thickness Avg (mm)'], bins = bins)
# store thickness counts in thickness accuracy dataframe
thickness_accuracy_df['OccurenceNumber'] = thickness_counts

print('Accuracy scores by thickenss bin')
print(thickness_accuracy_df)

#####################################################################################################################################
## Plotting

font_info = {'fontname': 'Arial', 'size':12}
tfont_info = {'fontname': 'Arial', 'size':14}

## CUSTOM COLOR MAP
def create_custom_colormap(colors):

    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
    return cmap

# Specify your hex colors
hex_colors = ['#FFFFFF','#FADE83', '#B1D78A', '#388968', '#276670', '#1E415D']

# Create the custom colormap
custom_cmap = create_custom_colormap(hex_colors)

#____________________________________________________________________________________________________________________
# FIGURE: Heat Map Confusion Matrix
fig,ax = plt.subplots(1,1, figsize = (5,4))

# Stage 1 subplot
avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
std_confusion_matrix = np.std(confusion_matrices, axis=0)

annot = np.array([f'{avg:.1f}\n±{std: .1f}' for avg, std in zip(avg_confusion_matrix.flatten(),std_confusion_matrix.flatten())])
annot = annot.reshape(avg_confusion_matrix.shape)
sns.heatmap(avg_confusion_matrix, annot=annot, fmt = '', cmap = custom_cmap, ax = ax, annot_kws=font_info)

ax.set_xticklabels(class_names, **font_info)
ax.set_yticklabels(class_names, **font_info)
ax.set_xlabel('Predicted Label', **tfont_info)
ax.set_ylabel('True Label', **tfont_info)

fig.savefig(f'{result_folder_path}/PCA_RF_confusion_matrix_{IR}_fig.pdf')


#_______________________________________________________________________________________________________________________
# FIGURE: Sample Importance Bar Chart
fig1, (ax11,ax12,ax13) = plt.subplots(1,3)
fig1.set_figheight(5)
fig1.set_figwidth(15)


# Color Characteristic Subplot--------------------------------------------
color_accuracy_df['Accuracy'].plot(kind='bar', ax = ax11, color= hex_colors[5])
ax11.set_xticklabels(color_accuracy_df.index, rotation=45, **font_info)
ax11.set_xlabel('Sample Color', **tfont_info)
ax11.tick_params(axis='y', labelsize=font_info['size'], labelfontfamily=font_info['fontname'], direction='in')
ax11.set_ylabel('Prediction Accuracy', **tfont_info)

# Increase space between the axes and the first and last bar
ax11.set_xlim(left=-0.8, right=len(color_accuracy_df)-0.2) 

# Adding value labels to the bars
i=0
for bar in ax11.patches:
    ax11.annotate(format(color_accuracy_df['OccurenceNumber'][i], '.0f'), 
                 (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                 ha='center', va='center', 
                 xytext=(0, 5), 
                 textcoords='offset points',
                 fontsize=font_info['size'],
                 color='black')
    i = i +1
ax11.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

# Transparency Subplot-------------------------------------------------------
transmission_accuracy_df['Accuracy'].plot(kind='bar', ax=ax12, color= hex_colors[3])
ax12.set_xticklabels(transmission_accuracy_df.index, rotation=45,**font_info)
ax12.set_xlabel('Sample Transparency', **tfont_info)
ax12.tick_params(axis='y', labelsize=font_info['size'], labelfontfamily=font_info['fontname'], direction='in')
#ax12.set_xlim(left=-0.8, right=len(transmission_accuracy_df)-0.2) 
# Adding value labels to the bars
i=0
for bar in ax12.patches:
    ax12.annotate(format(transmission_accuracy_df['OccurenceNumber'][i], '.0f'), 
                 (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                 ha='center', va='center', 
                 xytext=(0, 5), 
                 textcoords='offset points',
                 fontsize=font_info['size'],
                 color='black')
    i = i +1
ax12.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

# Thickness Subplot-----------------------------------------------------------
thickness_accuracy_df['Accuracy'].plot(kind='bar', ax=ax13, color = hex_colors[1])
ax13.set_xticklabels(thickness_accuracy_df.index, **font_info)
ax13.set_xlabel('Sample Thickness [mm]', **tfont_info)
ax13.tick_params(axis='y', labelsize=font_info['size'], labelfontfamily=font_info['fontname'], direction='in')
ax13.set_xlim(left=-1, right=len(thickness_accuracy_df)-0) 
# Adding value labels to the bars
i=0.00000001
for bar in ax13.patches:
    ax13.annotate(format(thickness_accuracy_df['OccurenceNumber'][i], '.0f'), 
                 (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                 ha='center', va='center', 
                 xytext=(0, 5), 
                 textcoords='offset points',
                 fontsize=font_info['size'],
                 color='black')
    i = i + increment

ax13.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

fig1.savefig(f'{result_folder_path}/PCA_RF_sample_charac_{IR}_fig.pdf')


#___________________________________________________________________________________________________________________
# FIGURE: Plotting of prediction probablity thresholds

# Preprocess data to find avgs and stds of all splits
avg_thresh_accuracies = np.mean(thresh_accuracies, axis=0)
std_thresh_accuracies = np.std(thresh_accuracies, axis=0)

avg_percentage_samples_lost = np.mean(percentage_samples_lost, axis=0)
std_percentage_samples_lost = np.std(percentage_samples_lost, axis=0)

# plot classification accuracy vs threshold 
fig2, ax_prob = plt.subplots(1,1)
ax_prob.errorbar(prob_threshold, avg_thresh_accuracies, yerr=std_thresh_accuracies, fmt='o', capsize=2)
ax_prob.set_xlabel('Model Probability Threshold', **tfont_info)
ax_prob.set_ylabel('Classification Accuracy', **tfont_info)
ax_prob.tick_params(axis='both', which='major',labelsize=font_info['size'], labelfontfamily=font_info['fontname'], direction='in')
ax_prob.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

# on same plot, plot percent of samples 'lost' vs threshold
ax2_prob = ax_prob.twinx()
ax2_prob.errorbar(prob_threshold[:-2], avg_percentage_samples_lost[:-2], yerr=std_percentage_samples_lost[:-2], fmt='s', color='g', capsize=2)
ax2_prob.set_ylabel('Percent of Samples "Lost"', **tfont_info, rotation = 270, labelpad=10)
ax2_prob.tick_params(axis='y', which='major',labelsize=font_info['size'], labelfontfamily=font_info['fontname'], direction='in')

fig2.savefig(f'{result_folder_path}/PCA_RF_prob_thresh_accuracies_{IR}_fig.pdf')
