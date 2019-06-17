# car_model_classification
Car model classification of Stanford car dataset

Deep learning technique: Transfer learning of pretrained VGG16 on IMAGENET dataset is used to classify car models using Stanford car dataset.

Dataset: Download the dataset from the Stanford_car_dataset weblink and copy train and test folders from car_data folder to this  forked git folder.  

Details of files: This repository mainly contains preprocessed train and test dataset, car_model_classification.py, preprocessing.py and plotting.py. preprocessing.py is used to crop the images to the given bounding boxes as given in the annotation files. car_model_classification.py is the code for transfer learning. plotting.py plots the accuracy vs epoch figure.

Requirements: Initial import packages in the car_model_classification.py gives an overview of the packages to be installed before using this repository. 
General requirements are : tensorflow, keras, matplotlib, numpy, scipy, csv

Step by step procedure to run the repository:

1> After downloading the dataset to this folder, read through the comments in the preprocessing.py code and run >> python preprocessing.py >> this will replace the existing train and test folder with cropped images.

2> After preprocessing, run >> python car_model_classification.py >> this will save accuracy_vs_epochs.csv and model file '.h5' to this folder.

3> Finally plot the accuracy vs epochs, run >> python plotting.py >>
