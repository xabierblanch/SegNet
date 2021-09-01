import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import *
from keras.models import load_model
from SegNet import *
from CNN_test import *
from utils import *

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

# Load the storage model
# model = load_model('models/SegNet_t1.h5')

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
history = model.fit(train_images, train_masks, batch_size=4 ,epochs=10)
test_loss, test_acc = model.evaluate(val_images, val_masks, batch_size=1)
test_masks = model.predict(train_images)

plt.imshow(train_images[5])
plt.imshow(train_masks[5], alpha=0.35)
plt.show()
plt.imshow(train_images[5])
plt.imshow(test_masks[5], alpha=0.35)
plt.show()

# Save the trained model as hdf5 file
model.save('models/SegNet_t1.h5')
