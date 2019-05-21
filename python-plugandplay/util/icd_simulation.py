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

def icd_simulation(z,y,h,sigw,lambd,K,filt_choice):
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  N = rows_hr*cols_hr
  # initialize cpp wrapper for icd
  icd_cpp = Pyicd(y,h,K,lambd,sigw);
  # read pre-trained model for pseudo-proximal map
  model_dir=os.path.join(os.getcwd(),'cnn')
  model_name = "model_sinc_sig60_realim_resnet"
  json_file = open(os.path.join(model_dir, model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights(os.path.join(model_dir,model_name+'.h5'))
  # iterative reconstruction
  v_icd = np.zeros((rows_hr,cols_hr))
  v_fft = copy.deepcopy(v_icd)
  v_cnn = copy.deepcopy(v_icd)
  #v=z
  print('itr      cost')
  forward_cost = []
  fft_cost = []
  cnn_cost = []
  for itr in range(20):
    print(itr)
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
    forward_cost.append(sum(sum((Gx_icd-y)**2))/(2*sigw*sigw) + sum(sum((x_icd-v_icd)**2))*lambd/2)
    fft_cost.append(sum(sum((Gx_fft-y)**2))/(2*sigw*sigw) + sum(sum((x_fft-v_fft)**2))*lambd/2)
    cnn_cost.append(sum(sum((Gx_cnn-y)**2))/(2*sigw*sigw) + sum(sum((x_cnn-v_cnn)**2))*lambd/2)
    v_fft = copy.deepcopy(x_fft)
    v_icd = copy.deepcopy(x_icd)
    v_cnn = copy.deepcopy(x_cnn)
    imsave('simulation_icd_itr'+str(itr)+'.png', np.clip(v_icd,0,1))
    imsave('simulation_fft_itr'+str(itr)+'.png', np.clip(v_fft,0,1))
    imsave('simulation_cnn_itr'+str(itr)+model_name+'.png', np.clip(v_cnn,0,1))
  # end ADMM recursive update
  plt.figure()
  plt.plot(list(range(forward_cost.__len__()))[5:],forward_cost[5:],label='ICD')  
  plt.plot(list(range(fft_cost.__len__()))[5:],fft_cost[5:],label='Fourier Decomposition')  
  plt.plot(list(range(cnn_cost.__len__()))[5:],cnn_cost[5:],label='Pseudo-proxmal map')  
  plt.xlabel('iteration')
  plt.ylabel('proximal map cost')
  plt.legend()
  plt.savefig('proximal_map_cost_'+model_name+filt_choice+'.png')
  err_img_icd = np.abs(x_icd-z)
  err_img_fft = np.abs(x_fft-z)
  err_img_cnn = np.abs(x_cnn-z)
  imsave('simulation_output_icd_'+filt_choice+'.png',np.clip(x_icd,0,1))
  imsave('simulation_output_fft_'+filt_choice+'.png',np.clip(x_fft,0,1))
  imsave('simulation_output_cnn_'+model_name+'.png',np.clip(x_cnn,0,1))
  imsave('err_img_icd_'+filt_choice+'.png',np.clip(err_img_icd,0,1))
  imsave('err_img_fft_'+filt_choice+'.png',np.clip(err_img_fft,0,1))
  imsave('err_img_cnn_'+model_name+'.png',np.clip(err_img_cnn,0,1))
  return

