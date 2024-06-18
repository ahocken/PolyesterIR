READ ME FIRST

These codes and files were used in the following manuscript:

Improving the optical sorting of polyester bioplastics via reflectance spectroscopy and machine learning classification techniques
Alexis Hocken, Sharona Huang, Eleanor Ng, Bradley D. Olsen

The following outlines the content within this record. Script files should be run in the order in which they are presented:

All spectral data is provided in the MIR Data and NIR Data folders. Please refer to supplementary spreadsheet provided with manuscript to see sample attributes such as thickness, color, and opacity.

Script for generating synthetic spectral data for scarce material classes

- SynSpectra.py
  
Scripts for compiling spectra data into a single file

- data_compile_NIR.py
  
- data_compile_MIR.py
  
Script to generate shuffled combined spectra data file and get subset indices

- data_split.py
  
Scripts to train and test ML models to classify material by spectral data inputs

- RF_torch_subset.py
  
- kNN_torch_subset.py
  
- PCA_RF_torch_subset.py
  
- PCA_kNN_torch_subset.py
  
Script to train and test RF model using a gradually reduced feature input set

- RF_reduced_feature_loop.py
  
Script to generate spectra with varying levels of synthetic Gaussian noise

- Residue_gaussian_noise.py
  
Script for compiling synthetically noised spectral data into single files

- data_compile_food_residue_NIR.py
  
- data_compile_food_residue_MIR.py
  
Script to generate shuffled combined spectra data file and get subset indices to match those in data_split.py

- food_residue_shuffle.py
  
Script to test RF and kNN models using synthetically noised spectral data to mimic food residue

- RF_kNN_food_residue.py
