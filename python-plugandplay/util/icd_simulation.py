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

def icd_simulation(z,y,h,sigw,lambd,K):
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  N = rows_hr*cols_hr
  # initialize cpp wrapper for icd
  icd_cpp = Pyicd(y,h,K,lambd,sigw);
  # use GGMRF as prior 
  np.random.seed(2019)
  x = np.random.rand(rows_hr, cols_hr)
  imsave('simulation_init.png', np.clip(x,0,1))
  # iterative reconstruction
  xtilde = np.zeros((rows_hr,cols_hr))
  print('itr      cost')
  forward_cost = []
  #rmse=[]
  for itr in range(21):
    # forward model optimization step
    Gx = construct_forward_model(x,K,h,0)
    forward_cost.append(sum(sum((Gx-y)**2))/(2*sigw*sigw) + sum(sum((x-xtilde)**2))*lambd/2)
    #rmse.append(sqrt(mean_squared_error(x,z)))
    #err_img = np.abs(x-z)
    #imsave('err_img_itr'+str(itr)+'.png',err_img)
    imsave('simulation_itr'+str(itr)+'.png', np.clip(x,0,1))
    # ICD update
    x = np.array(icd_cpp.update(x,xtilde))
    #print('icd time elapsed: ',toc-tic)
    print(itr,' ',forward_cost[-1])
  # end ADMM recursive update
  plt.plot(list(range(forward_cost.__len__())),forward_cost)
  plt.xlabel('iteration')
  plt.ylabel('proximal map cost')
  plt.savefig('proximal_map_cost.png')
  plt.figure()
  return np.array(x)

