import glob
from keras.applications.vgg16 import VGG16,decode_predictions
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pickle
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_dir = '/root/datasets/ILSVRC2017/Data/DET/train/ILSVRC2013_train/'
dict_name = '/root/datasets/DPM_VGG16_mixed4_train10_laplace0.1.dat'

dir_pool = ["n01503061","n02108089","n04330267","n04209133","n02802426","n02958343","n02132136","n02280649","n03179701","n03599486"]

def vgg_preprocess(im):
  x = copy.deepcopy(im*255.)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  return x
vggModel = VGG16(weights="imagenet")
yAv = []
v = []
epsil = []
mismatch_pool = []
np.random.seed(2019)
for subdir in dir_pool:
  fulldir = os.path.join(dataset_dir,subdir)
  filepool = glob.glob(os.path.join(fulldir,"*.JPEG"))
  n_samples = np.shape(filepool)[0]
  for filename in filepool[0:n_samples//4]:
    v_k = image.img_to_array(image.load_img(filename, target_size=(224, 224)))/255.
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
  
  for filename in filepool[n_samples//4:2*n_samples//4]:
    x_k = image.img_to_array(image.load_img(filename, target_size=(224, 224)))/255.
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

  for filename in filepool[n_samples*2//4:n_samples*3//4]:
    mismatch_pool.append(filename)
  for filename in filepool[n_samples*3//4:n_samples]:
    v_k = image.img_to_array(image.load_img(filename, target_size=(224, 224)))/255.
    epsil.append(np.zeros((224,224,3)))
    yAv.append(np.zeros((1,1000)))
    v.append(v_k)
mismatch_samples = np.shape(mismatch_pool)[0]
x_idx_list = np.random.choice(range(0,mismatch_samples), size=mismatch_samples, replace=False)
v_idx_list = np.random.choice(range(0,mismatch_samples), size=mismatch_samples, replace=False)
for (x_idx,v_idx) in zip(x_idx_list,v_idx_list):
  x_k = image.img_to_array(image.load_img(mismatch_pool[x_idx], target_size=(224, 224)))/255.
  v_k = image.img_to_array(image.load_img(mismatch_pool[v_idx], target_size=(224, 224)))/255.
  epsil_k = x_k - v_k
  y = vggModel.predict(vgg_preprocess(x_k))
  Av = vggModel.predict(vgg_preprocess(v_k))
  y_Av_k = y-Av
  yAv.append(y_Av_k)
  v.append(v_k)
  epsil.append(epsil_k)
yAv = np.reshape(yAv,(-1,1000))
print("shape of v: ",np.shape(v))
print("shape of epsil: ",np.shape(epsil))
print("shape of y-Av: ",np.shape(yAv))
dataset = {'epsil':epsil,'y_Av':yAv, 'v':v}
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
