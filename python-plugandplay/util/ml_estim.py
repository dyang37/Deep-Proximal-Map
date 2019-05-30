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

def ml_estimate(y,h,sigk,sig,K):
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  N = rows_hr*cols_hr
  lambd = 1./(sig*sig)
  # initialize cpp wrapper for icd
  icd_cpp = Pyicd(y,h,K,lambd,sigk);
  # read pre-trained model for pseudo-proximal map
  model_dir=os.path.join(os.getcwd(),'cnn')
  model_name = "model_sinc_noisy_simple_hr"
  json_file = open(os.path.join(model_dir, model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights(os.path.join(model_dir,model_name+'.h5'))
  # iterative reconstruction
  v_icd = np.zeros((rows_hr,cols_hr))
  v_cnn = copy.deepcopy(v_icd) 
  F_y = np.random.rand(rows_hr, cols_hr)
  ml_cost_cnn = []
  ml_cost_icd = []
  for itr in range(20):
    # ICD iterative update
    x_icd = np.random.rand(rows_hr,cols_hr)
    for icd_itr in range(10):
      x_icd = np.array(icd_cpp.update(x_icd, v_icd))
    v_icd = x_icd
    fv = construct_forward_model(v_cnn, K, h, 0)
    H = pseudo_prox_map(np.subtract(y,fv), model)
    v_cnn = np.add(v_icd, H)
    y_icd = construct_forward_model(v_icd, K, h, 0)
    y_cnn = construct_forward_model(v_cnn, K, h, 0)
    ml_cost_cnn.append(((y-y_cnn)**2).mean(axis=None))
    ml_cost_icd.append(((y-y_icd)**2).mean(axis=None))
  # check if H(y) = F_y(0)
  mse = ((v_icd-v_cnn)**2).mean(axis=None)
  mse_y_gd = ((y-y_cnn)**2).mean(axis=None)
  mse_y = ((y_icd-y_cnn)**2).mean(axis=None)
  print('pixelwise mse value: ',mse)
  print('pixelwise mse value for y between cnn and groundtruth: ',mse_y_gd)
  print('pixelwise mse value for y between cnn and icd: ',mse_y)
  
  # cost function plot
  plt.plot(list(range(ml_cost_cnn.__len__())),ml_cost_cnn,label="deep prox map")
  plt.plot(list(range(ml_cost_icd.__len__())),ml_cost_icd,label="icd")
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.xlabel('iteration')
  plt.ylabel('ML cost $||Y-Ax||^2$')
  
  err_img = np.subtract(v_icd,v_cnn) + 0.5
  err_y = np.subtract(y,y_cnn) + 0.5
  # save output images
  imsave('diff_exp.png',np.clip(err_img,0,1))
  imsave('diff_y.png',np.clip(err_y,0,1))
  imsave('Hy.png',np.clip(Hy,0,1))
  imsave('F_y0.png',np.clip(F_y,0,1))
  return

