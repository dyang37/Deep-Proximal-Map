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

def cnn_exp(z,y,h,sigw,lambd,K,filt_choice):
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  N = rows_hr*cols_hr
  # initialize cpp wrapper for icd
  icd_cpp = Pyicd(y,h,K,lambd,sigw);
  # read pre-trained model for pseudo-proximal map
  model_dir=os.path.join(os.getcwd(),'cnn')
  model_name = "model_sinc_sig60_realim"
  print('using model:',model_name)
  json_file = open(os.path.join(model_dir, model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights(os.path.join(model_dir,model_name+'.h5'))
  # use GGMRF as prior 
  np.random.seed(2019)
  # iterative reconstruction
  v_icd = z
  #v_icd = np.zeros((rows_hr,cols_hr))
  v_fft = copy.deepcopy(v_icd)
  v_cnn = copy.deepcopy(v_icd)
  #Gx_cnn = construct_forward_model(x_cnn,K,h,0)
  x_icd = np.random.rand(rows_hr, cols_hr)
  x_fft = forward_model_optim(x_icd,v_fft,y,h,K,lambd,sigw)
  fv = construct_forward_model(v_cnn, K, h, 0)
  x_cnn = np.add(pseudo_prox_map(fv, y, v_cnn, model), v_cnn)
  # ICD iterative update
  for icd_itr in range(10):
    x_icd = np.array(icd_cpp.update(x_icd,v_icd))
    
  Gx_fft = construct_forward_model(x_fft,K,h,0)
  Gx_icd = construct_forward_model(x_icd,K,h,0)
  Gx_cnn = construct_forward_model(x_cnn,K,h,0)
  forward_cost = sum(sum((Gx_icd-y)**2))/(2*sigw*sigw) + sum(sum((x_icd-v_icd)**2))*lambd/2
  fft_cost = sum(sum((Gx_fft-y)**2))/(2*sigw*sigw) + sum(sum((x_fft-v_fft)**2))*lambd/2
  cnn_cost = sum(sum((Gx_cnn-y)**2))/(2*sigw*sigw) + sum(sum((x_cnn-v_cnn)**2))*lambd/2
  imsave('denoise_icd'+'.png', np.clip(x_icd,0,1))
  imsave('denoise_fft'+'.png', np.clip(x_fft,0,1))
  imsave('denoise_cnn'+'.png', np.clip(x_cnn,0,1))
  # end ADMM recursive update
  print('icd cost:',forward_cost)
  print('fft cost:', fft_cost)
  print('cnn cost:', cnn_cost)
  
  err_img_icd = np.abs(x_icd-z)
  err_img_fft = np.abs(x_fft-z)
  err_img_cnn = np.abs(x_cnn-z)
  imsave('denoise_err_img_icd_'+filt_choice+'.png',np.clip(err_img_icd,0,1))
  imsave('denoise_err_img_fft_'+filt_choice+'.png',np.clip(err_img_fft,0,1))
  imsave('denoise_err_img_cnn_'+filt_choice+'.png',np.clip(err_img_cnn,0,1))
  return

