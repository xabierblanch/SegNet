import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def create_training_data(DATADIR, IMG_SIZE):
  training_data = []
  train_images = []
  train_masks = []
  path = os.path.join(DATADIR, 'img')
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = cv2.normalize(img_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    train_images.append([img_array])
    mask_array = cv2.imread(os.path.join(DATADIR, 'label', img), cv2.IMREAD_GRAYSCALE)
    mask_array = cv2.resize(mask_array, (IMG_SIZE, IMG_SIZE))
    mask_array = cv2.normalize(mask_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_32F)
    train_masks.append([mask_array])
    training_data.append([img_array, mask_array])

  train_images = np.array(train_images)
  train_images = train_images.reshape(len(os.listdir(path)), IMG_SIZE, IMG_SIZE, 3)
  train_masks = np.array(train_masks)
  train_masks = train_masks.reshape(len(os.listdir(path)), IMG_SIZE, IMG_SIZE, 1)

  return train_images, train_masks, training_data

def show_images(id, data):
  plt.imshow(data[id][0])  # interpolation='none'
  plt.xlabel('Values = Min: {:.1f}  Max: {:.1f}'.format(data[id][0].min(), data[id][0].max()))
  plt.ylabel('Image size = ' + str(data[id][0].shape))
  plt.title('RAW Image (id: ' + str(id) + ')')
  plt.show()
  plt.imshow(data[id][1], cmap='jet')  # interpolation='none'
  plt.title('MASK Image (id: ' + str(id) + ')')
  plt.xlabel('Values = Min: {:.1f}  Max: {:.1f}'.format(data[id][1].min(), data[id][1].max()))
  plt.ylabel('Image size = ' + str(data[id][1].shape))
  plt.show()
  plt.imshow(data[id][0])  # interpolation='none'
  plt.imshow(data[id][1], cmap='jet', alpha=0.35)  # interpolation='none'
  plt.title('MIX Image (id: ' + str(id) + ')')
  plt.show()