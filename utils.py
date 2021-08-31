import cv2
import os

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