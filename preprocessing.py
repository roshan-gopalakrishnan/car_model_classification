""" Developer: Roshan Gopalakrishnan
    Contact: roshan.gopalakrishnan@gmail.com

    Description:
    This code crops the train and test - Stanford_car_dataset with respect to
    the bounding box details provided in the annotation files inside devkit.

    Annotation files:
    Cars_train_annos.mat and cars_test_annos.mat
    These files can be downloaded from the Stanford_car_dataset weblink

    Problem encountered:
    One of the folder, "Ram C" among the subfolders in train in car_data,
    (hierarchy: car_data - train = Ram C ), has another subfolder, while other
    subfolders in train contain the image files, hence the looping statement in
    this code was showing error, so I removed that folder and then later separately
    cropped the images within the "Ram C" folder.
    Note: If you want to use this code on freshly downloaded car_data zip file
    from Stanford_car_dataset weblink, please aware of this problem in this code.
    If you remove the "Ram C" folders from train and test folders you will not
    encounter any errors.
"""

""" Import packages """

import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.misc
import numpy as np
import keras
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input
from scipy.io import loadmat

""" Load and print .mat file """

annots = loadmat('devkit/cars_train_annos.mat')
print ('Type of mat file:', type(annots))
print ('Dictionary keys:', annots.keys())
print ('annotations details', type(annots['annotations']), annots['annotations'].shape)
# print 'annotations version', type(annots['__version__']), annots['__version__']
# print 'annotations headers', type(annots['__header__']), annots['__header__']
# print 'annotations global', type(annots['__globals__']), np.shape(annots['__globals__'])
# print 'subdirectories:', type(annots['annotations'][0][0]), annots['annotations'][0][0].shape
# print(annots['annotations'][0])
print(annots['annotations'][0][0])

""" Printing the headers and flats as given in README doc """

print (annots['annotations'][0][0]['bbox_x1'], annots['annotations'][0][0]['fname'])
print (annots['annotations'][0][0]['bbox_x1'].flat[0])
print ([item.flat[0] for item in annots['annotations'][0][0]])

""" Flattening and processing of train data """

data = [[row.flat[0] for row in line] for line in annots['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
df_train = pd.DataFrame(data, columns=columns)
print (df_train.head())

""" Cropping train data with bounding box measurements """

print ("Cropping train data with bounding box measurements")
for root, subdirs, files in os.walk('./car_data/train'):
#for root, subdirs, files in os.walk('./car_data/train/Ram C'):
    for file in files:
        for index, row in df_train.iterrows():
            if (file == row['fname']):
                print (index, row['bbox_y1'], row['bbox_y2'], row['bbox_x1'], row['bbox_x2'], row['class'], row['fname'])
                print (os.path.join(root,file))
                image = plt.imread(os.path.join(root,file))
                print ('Original size', np.shape(image))
                if (np.shape(image)==3):
                    image = image[row['bbox_y1']:row['bbox_y2'] , row['bbox_x1']:row['bbox_x2'], :]
                else:
                    image = image[row['bbox_y1']:row['bbox_y2'] , row['bbox_x1']:row['bbox_x2']]
                image = image_utils.load_img(image, target_size=(224, 224))
                image = image_utils.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                print ('Cropped size', np.shape(image))
                p = os.path.join(root,file).split("/")
                dest = os.path.join('/home/roshan/Desktop',p[1], p[2], p[3], p[4])
                #dest = os.path.join('/home/roshan/Desktop',p[1], p[2], p[3], p[4], p[5])
                print (dest)
                scipy.misc.imsave(dest, image)

""" Flattening and processing of train data """

annots = loadmat('devkit/cars_test_annos.mat')
data = [[row.flat[0] for row in line] for line in annots['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
df_test = pd.DataFrame(data, columns=columns)
print (df_test.head())

""" Cropping test data with bounding box measurements """

print ("Cropping test data with bounding box measurements")
# for root, subdirs, files in os.walk('./car_data/test/Ram C'):
for root, subdirs, files in os.walk('./car_data/test'):
    for file in files:
        for index, row in df_test.iterrows():
            if (file == row['fname']):
                print (index, row['bbox_y1'], row['bbox_y2'], row['bbox_x1'], row['bbox_x2'], row['fname'])
                print (os.path.join(root,file))
                image = plt.imread(os.path.join(root,file))
                print ('Original size', np.shape(image))
                if (np.shape(image)==3):
                    image = image[row['bbox_y1']:row['bbox_y2'] , row['bbox_x1']:row['bbox_x2'], :]
                else:
                    image = image[row['bbox_y1']:row['bbox_y2'] , row['bbox_x1']:row['bbox_x2']]
                image = image_utils.load_img(image, target_size=(224, 224))
                image = image_utils.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                print ('Cropped size', np.shape(image))
                p = os.path.join(root,file).split("/")
                dest = os.path.join('/home/roshan/Desktop',p[1], p[2], p[3], p[4])
                # dest = os.path.join('/home/roshan/Desktop',p[1], p[2], p[3], p[4], p[5])
                print (dest)
                scipy.misc.imsave(dest, image)


""" Plotting of single example from train data """
# if you want to see the plot just uncomment this section below
"""
file = './car_data/train/Acura Integra Type R 2001/00198.jpg'
dirs = file.split("/")
print dirs
print dirs[-1]

fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(plt.imread(file))

for index, row in df_train.iterrows():
    if (dirs[-1] == row['fname']):
        print row['bbox_y1'], row['bbox_y2'], row['bbox_x1'], row['bbox_x2'], row['class'], row['fname']
        image = plt.imread(file)
        image = image[row['bbox_y1']:row['bbox_y2'] , row['bbox_x1']:row['bbox_x2'], :]
        scipy.misc.imsave(file, image)


ax1 = fig.add_subplot(122)
ax1.imshow(image)
plt.show()
"""
