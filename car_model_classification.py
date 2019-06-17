""" Developer: Roshan Gopalakrishnan
    Contact: roshan.gopalakrishnan@gmail.com

    Description:
    This code is to classify the make and model of Stanford_car_dataset.
    Deep learning - transfer learning technique is used here.

    Dataset preprocessing:
    Dataset preprocessing is done in the preprocessing.py code in this root.
    You may first preprocess the dataset before running this code.

"""

""" Import packages """

import os
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

""" Set the GPU enviornment """
# change the visible devices as needed

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"


""" Allow better GPU memory allocation """

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

print ('Experiment: Transfer Learning on Stanford Car dataset using VGGnet16')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc    = []
        self.val_loss = []
        self.val_acc    = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))


img_width, img_height = 224, 224
train_data_dir = 'train'
validation_data_dir = 'test'
epochs = 100
batch_size = 32
num_classes = 196

if K.image_data_format() == 'channels_first':
    image_input_shape = (3, img_width, img_height)
else:
    image_input_shape = (img_width, img_height, 3)
print('input shape:', image_input_shape)

# Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=image_input_shape, pooling='avg')

# Freeze all layers but not top 4
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# compile the model with a adam optimizer
# and a very slow learning rate.
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-6)
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# fine-tune the model
history = LossHistory()
log_file = CSVLogger('accuracy_vs_epochs.csv', append=True, separator=';')
parallel_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    callbacks=[history, log_file])


# Save the Model
model.save('./Car_TL_VGG16.h5')
