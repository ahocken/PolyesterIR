READ ME FIRST

These codes and files were used in the following manuscript:

Improving the optical sorting of polyester bioplastics via reflectance spectroscopy and machine learning classification techniques
Alexis Hocken, Sharona Huang, Eleanor Ng, Bradley D. Olsen

The following outlines the content within this record. Script files should be run in the order in which they are presented:
•	Script for generating synthetic spectral data for scarce material classes
  o	SynSpectra.py
•	Scripts for compiling spectra data into a single file
  o	data_compile_NIR.py
  o	data_compile_MIR.py
•	Script to generate shuffled combined spectra data file and get subset indices
  o	data_split.py
•	Scripts to train and test ML models to classify material by spectral data inputs
  o	RF_torch_subset.py
  o	kNN_torch_subset.py
  o	PCA_RF_torch_subset.py
  o	PCA_kNN_torch_subset.py
•	Script to train and test RF model using a gradually reduced feature input set
  o	RF_reduced_feature_loop.py
•	Script to generate spectra with varying levels of synthetic Gaussian noise
  o	Residue_gaussian_noise.py
•	Script for compiling synthetically noised spectral data into single files
  o	data_compile_food_residue_NIR.py
  o	data_compile_food_residue_MIR.py
•	Script to generate shuffled combined spectra data file and get subset indices to match those in data_split.py
  o	food_residue_shuffle.py
•	Script to test RF and kNN models using synthetically noised spectral data to mimic food residue
  o	RF_kNN_food_residue.py
