import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from SegNet import *
from CNN_test import *
from utils import *

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

#main_parameters
DATADIR = 'C:/Users/XBG-KIWA/Documents/01_GITHUB_(XBG)/SegNet/data'
IMG_SIZE = 64

#load images and masks -> resize and normalise
train_images, train_masks, training_data = create_training_data(DATADIR, IMG_SIZE)

#chech images
# show_images(25, training_data)

#load CNN SegNet
# model = SegNet(IMG_SIZE,IMG_SIZE)
# model.summary()
# # model.compile(optimizer='sgd', loss = tf.keras.losses.SparseCategoricalCrossentropy(),
# #               metrics=['accuracy'])
# #
# # pred_mask = model.predict(train_images[0])


#load CNN SegNet
model=CNN_test(IMG_SIZE)
model.summary()
opt = SGD(lr=0.001, momentum=0.9, decay=0.0005)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(train_images, train_masks, batch_size=1 ,epochs=5)

# pred_mask = model.predict(train_images)
# plt.imshow(pred_mask[0])
# plt.show



