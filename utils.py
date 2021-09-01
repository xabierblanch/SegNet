import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def create_datasets(DIR, IMG_SIZE):
  images = []
  masks = []
  path = os.path.join(DIR, 'img')
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = cv2.normalize(img_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    images.append([img_array])
    try:
      mask_array = cv2.imread(os.path.join(DIR, 'label', img), cv2.IMREAD_GRAYSCALE)
      mask_array = cv2.resize(mask_array, (IMG_SIZE, IMG_SIZE))
      mask_array = cv2.normalize(mask_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_32F)
      masks.append([mask_array])
    except:
      continue
  if images:
    train_images = np.array(images)
    train_images = train_images.reshape(len(os.listdir(os.path.join(DIR, 'img'))), IMG_SIZE, IMG_SIZE, 3)
  else:
    train_images = []
  if masks:
    train_masks = np.array(masks)
    train_masks = train_masks.reshape(len(os.listdir(os.path.join(DIR, 'label'))), IMG_SIZE, IMG_SIZE, 1)
  else:
    train_masks = []
  return train_images, train_masks

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