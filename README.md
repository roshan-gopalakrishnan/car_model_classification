# car_model_classification
Car model classification of Stanford car dataset

Deep learning technique: Transfer learning of pretrained VGG16 on IMAGENET dataset is used to classify car models using Stanford car dataset.

This repository mainly contains preprocessed train and test dataset, car_model_classification.py, preprocessing.py and plotting.py

preprocessing.py is used to crop the images to the given bounding boxes as given in the annotation files. 
car_model_classification.py is the code for transfer learning.
plotting.py plots the accuracy vs epoch figure.

Requirements: Initial import packages in the car_model_classification.py gives an overview of the packages to be installed before using this repository. 

General requirements are : tensorflow, keras, matplotlib, numpy, scipy, csv
