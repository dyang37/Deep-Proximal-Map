import numpy as np
import sys
import os
from math import sqrt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "./denoisers/DnCNN"))
from skimage.io import imsave
from scipy.misc import imresize
from construct_forward_model import construct_forward_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from icdwrapper import Pyicd
import timeit
import copy
from sklearn.metrics import mean_squared_error
from forward_model_optim import forward_model_optim
from dncnn import pseudo_prox_map
from keras.models import model_from_json

def deep_proximal_map_exp(y,h,sigk,sig,K):
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  N = rows_hr*cols_hr
  lambd = 1./(sig*sig)
  # initialize cpp wrapper for icd
  icd_cpp = Pyicd(y,h,K,lambd,sigk);
  # read pre-trained model for pseudo-proximal map
  model_dir=os.path.join(os.getcwd(),'cnn')
  model_name = "model_sinc_noisy_linear"
  json_file = open(os.path.join(model_dir, model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights(os.path.join(model_dir,model_name+'.h5'))
  # iterative reconstruction
  v = np.zeros((rows_hr,cols_hr))
  F_y = np.random.rand(rows_hr, cols_hr)
  # ICD iterative update
  for icd_itr in range(10):
    F_y = np.array(icd_cpp.update(F_y, v))
  
  Hy = pseudo_prox_map(y, model)
  print('shape of deep proximal map output:',np.shape(Hy))
  # check if H(y) = F_y(0)
  mse = ((F_y-Hy)**2).mean(axis=None)
  print('pixelwise mse value: ',mse)
  err_img = np.subtract(F_y,Hy) + 0.5
  imsave('diff_exp.png',np.clip(err_img,0,1))
  imsave('Hy.png',np.clip(Hy,0,1))
  imsave('F_y0.png',np.clip(F_y,0,1))
  return

