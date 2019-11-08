import glob
import cv2
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import pickle

dataset = []
for filename in glob.glob('/root/datasets/Data/DET/test/*.JPEG')[:5000]:
  image = image_utils.load_img(filename, target_size=(224, 224))
  image = image_utils.img_to_array(image)
  image = preprocess_input(image)
  dataset.append(image)
print("dataset shape:",np.shape(dataset))
dict_name = '/root/datasets/ILSVRC_test_small.dat'
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
