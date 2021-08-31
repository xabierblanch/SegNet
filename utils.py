import cv2
import os
import matplotlib.pyplot as plt

def create_training_data(DATADIR, CATEGORIES, IMG_SIZE):
  training_data = []
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
      try:
        if category == 'img':
          img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
          img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
          img_array = cv2.normalize(img_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        elif category == 'label':
          mask_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
          mask_array = cv2.resize(mask_array, (IMG_SIZE, IMG_SIZE))
          mask_array = cv2.normalize(mask_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_32F)
        training_data.append([img_array, mask_array])
      except Exception as e:
        pass
  return training_data

def show_image(id, data):
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