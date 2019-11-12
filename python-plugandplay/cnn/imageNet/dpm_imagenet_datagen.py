import glob
import cv2
from keras.preprocessing import image as image_utils
from keras.applications import VGG16
import numpy as np
import pickle
import copy
import os

dataset_dir = '/root/datasets/ILSVRC/Data/DET/train/ILSVRC2013_train/'
dict_name = '/root/datasets/DPM_VGG16_train10_laplace0.1.dat'
n_cate = 40

dir_pool = ["n01503061","n02108089","n04330267","n04209133","n02802426","n02958343","n02132136","n02280649","n03179701","n03599486"]

def vgg_preprocess(im):
  image = copy.deepcopy(im*255.)
  image[:,:,0] -= 103.939
  image[:,:,1] -= 116.779
  image[:,:,2] -= 123.68
  image = np.expand_dims(image, axis=0)
  return image
vggModel = VGG16(weights="imagenet")
yAv = []
v = []
epsil = []

for subdir in dir_pool:
  fulldir = os.path.join(dataset_dir,subdir)
  filepool = glob.glob(os.path.join(fulldir,"*.JPEG"))
  n_samples = np.shape(filepool)[0]
  for filename in filepool[0:n_samples//4]:
    v_k = cv2.resize(cv2.imread(filename), (224, 224)).astype(np.float32)/255.
    #sig = np.random.uniform(0,0.3)
    sig = min(abs(np.random.laplace(scale=0.1)), 0.5)
    epsil_k = np.random.normal(0,sig,v_k.shape)
    x_k = v_k + epsil_k
    y = vggModel.predict(vgg_preprocess(x_k))
    Av = vggModel.predict(vgg_preprocess(v_k))
    y_Av_k = y-Av
    yAv.append(y_Av_k)
    v.append(v_k)
    epsil.append(epsil_k)

  for filename in filepool[n_samples//4:n_samples*2//4]:
    x_k = cv2.resize(cv2.imread(filename), (224, 224)).astype(np.float32)/255.
    #sig = np.random.uniform(0,0.3)
    sig = min(abs(np.random.laplace(scale=0.1)), 0.5)
    epsil_k = np.random.normal(0,sig,x_k.shape)
    v_k = x_k - epsil_k
    y = vggModel.predict(vgg_preprocess(x_k))
    Av = vggModel.predict(vgg_preprocess(v_k))
    y_Av_k = y-Av
    yAv.append(y_Av_k)
    v.append(v_k)
    epsil.append(epsil_k)

  x_idx_list = np.random.choice(range(n_samples*2//4,n_samples*3//4), size=n_samples//4, replace=False)
  v_idx_list = np.random.choice(range(n_samples*2//4,n_samples*3//4), size=n_samples//4, replace=False)
  for (x_idx,v_idx) in zip(x_idx_list,v_idx_list):
    x_k = cv2.resize(cv2.imread(filepool[x_idx]), (224, 224)).astype(np.float32)/255.
    v_k = cv2.resize(cv2.imread(filepool[v_idx]), (224, 224)).astype(np.float32)/255.
    epsil_k = x_k - v_k
    y = vggModel.predict(vgg_preprocess(x_k))
    Av = vggModel.predict(vgg_preprocess(v_k))
    y_Av_k = y-Av
    yAv.append(y_Av_k)
    v.append(v_k)
    epsil.append(epsil_k)

  for filename in filepool[n_samples*3//4:n_samples]:
    v_k = cv2.resize(cv2.imread(filename), (224, 224)).astype(np.float32)/255.
    epsil.append(np.zeros((224,224,3)))
    yAv.append(np.zeros((1,1000)))
    v.append(v_k)

yAv = np.reshape(yAv,(-1,1000))
print("shape of v: ",np.shape(v))
print("shape of epsil: ",np.shape(epsil))
print("shape of y-Av: ",np.shape(yAv))
dataset = {'epsil':epsil,'y_Av':yAv, 'v':v}
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
