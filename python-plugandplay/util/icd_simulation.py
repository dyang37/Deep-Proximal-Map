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

def icd_simulation(z,y,h,sigw,lambd,K,filt_choice):
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  N = rows_hr*cols_hr
  # initialize cpp wrapper for icd
  icd_cpp = Pyicd(y,h,K,lambd,sigw);
  # use GGMRF as prior 
  np.random.seed(2019)
  x = np.random.rand(rows_hr, cols_hr)
  x_fft = copy.deepcopy(x)
  # iterative reconstruction
  xtilde = np.zeros((rows_hr,cols_hr))
  #xtilde=z
  print('itr      cost')
  forward_cost = []
  fft_cost = []
  for itr in range(21):
    # forward model optimization step
    Gx = construct_forward_model(x,K,h,0)
    Gx_fft = construct_forward_model(x_fft,K,h,0)
    forward_cost.append(sum(sum((Gx-y)**2))/(2*sigw*sigw) + sum(sum((x-xtilde)**2))*lambd/2)
    fft_cost.append(sum(sum((Gx_fft-y)**2))/(2*sigw*sigw) + sum(sum((x_fft-xtilde)**2))*lambd/2)
    #imsave('err_img_itr'+str(itr)+'.png',err_img)
    #imsave('simulation_itr'+str(itr)+'.png', np.clip(x,0,1))
    # ICD update
    x = np.array(icd_cpp.update(x,xtilde))
    x_fft = forward_model_optim(x_fft,xtilde,y,h,K,lambd,sigw)
  # end ADMM recursive update
  plt.figure()
  plt.plot(list(range(forward_cost.__len__()))[5:],forward_cost[5:],label='ICD')  
  plt.plot(list(range(fft_cost.__len__()))[5:],fft_cost[5:],label='Fourier Decomposition')  
  plt.xlabel('iteration')
  plt.ylabel('proximal map cost')
  plt.legend()
  plt.savefig('proximal_map_cost_'+filt_choice+'.png')
  err_img = np.abs(x-z)
  err_img_fft = np.abs(x_fft-z)
  imsave('simulation_output_icd_'+filt_choice+'.png',np.clip(x,0,1))
  imsave('simulation_output_fft_'+filt_choice+'.png',np.clip(x_fft,0,1))
  imsave('err_img_icd_'+filt_choice+'.png',np.clip(err_img,0,1))
  imsave('err_img_fft_'+filt_choice+'.png',np.clip(err_img_fft,0,1))
  return

