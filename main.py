import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import *
from SegNet import *
from CNN_test import *
from utils import *

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

#main_parameters
TRAIN_DIR = 'C:/Users/XBG-KIWA/Documents/01_GITHUB_(XBG)/SegNet/data/train'
VAL_DIR = 'C:/Users/XBG-KIWA/Documents/01_GITHUB_(XBG)/SegNet/data/val'
TEST_DIR = 'C:/Users/XBG-KIWA/Documents/01_GITHUB_(XBG)/SegNet/data/test'
IMG_SIZE = 128

#load images and masks -> resize and normalise
train_images, train_masks = create_datasets(TRAIN_DIR, IMG_SIZE)
val_images, val_masks = create_datasets(VAL_DIR, IMG_SIZE)
test_images, test_masks = create_datasets(TEST_DIR, IMG_SIZE)

#chech images
# show_images(25, training_data)

#load CNN SegNet
model = SegNet(IMG_SIZE,IMG_SIZE)
model.summary()
opt = SGD(lr=0.001, momentum=0.9, decay=0.0005)
model.compile(optimizer=opt, loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_masks, batch_size=5 ,epochs=5)

test_loss, test_acc = model.evaluate(train_images, train_masks, batch_size=1)

pred_mask = model.predict(train_images)

plt.imshow(train_masks[43])
plt.imshow(pred_mask[43], alpha=0.35)
plt.show()



TRAIN_DIR = 'C:/Users/XBG-KIWA/Documents/01_GITHUB_(XBG)/SegNet/data'
VALIDATION_DIR = './datasets/validation'
