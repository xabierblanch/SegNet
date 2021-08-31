import tensorflow as tf
from tensorflow import keras
from SegNet import *
from CNN_test import *
from utils import *

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

#main_parameters
DATADIR = 'C:/Users/XBG-KIWA/Documents/01_GITHUB_(XBG)/SegNet/data'
IMG_SIZE = 512

#load images and masks -> resize and normalise
train_images, train_masks, training_data = create_training_data(DATADIR, IMG_SIZE)

#chech images
# show_images(25, training_data)

#load CNN SegNet
# model = SegNet(IMG_SIZE,IMG_SIZE)
# model.summary()
# model.compile(optimizer='sgd', loss = tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])
#
# pred_mask = model.predict(train_images[0])


#load CNN SegNet
model=CNN_test()
model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#
# model.fit(train_images, train_masks, epochs=5)

