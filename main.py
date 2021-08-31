import tensorflow as tf
from tensorflow import keras
from SegNet import SegNet
from utils import *

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

#main_parameters
DATADIR = 'C:/Users/XBG-KIWA/Documents/01_GITHUB_(XBG)/SegNet/data'
CATEGORIES = ["img", "label"]
IMG_SIZE = 512

#load images and masks -> resize and normalise
data = create_training_data(DATADIR,CATEGORIES, IMG_SIZE)

#chech images
show_image(5, data)
