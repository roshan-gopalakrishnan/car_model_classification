# car_model_classification
Image classification of Stanford car dataset

I have used transfer learning of pretrained VGG16 on IMAGENET dataset to classify car models using Stanford car dataset.

This repository mainly contains preprocessed train and test dataset, car_model_classification.py, preprocessing.py and plotting.py

preprocessing.py is used to crop the images to the given bounding boxes in the annotation files. It also does the basic preprocessing needed for training and testing images.
car_model_classification.py is the code for transfer learning.
plotting.py plots the accuracy vs epoch figure.

Requirements: Initial import packages in the car_model_classification.py gives an overview of the packages to be installed before using this repository. 
General requirements are :
tensorflow
keras
matplotlib
numpy
scipy
csv
