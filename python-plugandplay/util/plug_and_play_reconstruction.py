import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imsave
from keras.models import  model_from_json
import copy
from dncnn import cnn_denoiser, pseudo_prox_map
from construct_forward_model import construct_forward_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from icdwrapper import Pyicd

# This function performs ADMM iterative reconstruction for image super resolution problem
# hr_img: ground_truth image. Only used for evaluation purpose
# y: low resolution input image
# h: anti-aliasing filter used in forward model
# sigw: noise std-deviation
# K: down/up sampling factor
# denoiser: choice for image denoising algorithm
#           0: DnCNN
#           1: Total Variation
#           2: Non-local Mean
# return value: x, the reconstructed high resolution image


def plug_and_play_reconstruction(hr_img,y,h,sigw,lambd,K,optim_method,filt_choice):
  # load pre-trained denoiser model
  if optim_method == 0:
    output_dir = os.path.join(os.getcwd(),'../results/pnp_output/pmap/')
  else:
    output_dir = os.path.join(os.getcwd(),'../results/pnp_output/icd/')
  denoiser_dir=os.path.join(os.getcwd(),'../denoisers/DnCNN')
  json_file = open(os.path.join(denoiser_dir,'model.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  denoiser_model = model_from_json(loaded_model_json)
  denoiser_model.load_weights(os.path.join(denoiser_dir,'model.h5'))
  
  # load pre-trained deep proximal map model
  pmap_dir=os.path.join(os.getcwd(),'cnn')
  pmap_model_name = "model_sinc_noisy_simple_hr"
  json_file = open(os.path.join(pmap_dir, pmap_model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  pmap_model = model_from_json(loaded_model_json)
  pmap_model.load_weights(os.path.join(pmap_dir,pmap_model_name+'.h5'))
  
  # initialize cpp wrapper for icd
  icd_wrapper = Pyicd(y,h,K,lambd,sigw);
  
  
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  
  # plug and play cost function
  g = np.array([[1/12,1/6,1/12],[1/6,0,1/6],[1/12,1/6,1/12]])
  p = 2 #### power param for GGMRF
  # estimate sigx
  sigx = 0
  for i in range(rows_hr):
    for j in range(cols_hr):
      for di in range(-1,2):
        for dj in range(-1,2):
          if(i+di>=0 and i+di<rows_hr and j+dj>=0 and j+dj<cols_hr):
            sigx += g[di+1,dj+1] * abs(hr_img[i,j]-hr_img[i+di,j+dj])**p
            # divide by 2 because we counted each clique twice
  sigx /= 2.
  sigx = (sigx/(rows_hr*cols_hr))**(1/p)
  print('estimated GGMRF sigma = ',sigx)

  x = np.random.rand(rows_hr, cols_hr)
  v = np.random.rand(rows_hr, cols_hr)
  u = np.zeros((rows_hr, cols_hr))
  # iterative reconstruction
  admm_cost = []
  residual_x = []
  residual_v = []
  for itr in range(20):
    # forward step
    x_old = copy.deepcopy(x)
    v_old = copy.deepcopy(v)
    vtilde = np.subtract(x,u)
    v = optimization_wrapper(v,vtilde,y,h,K,itr,optim_method, pmap_model, icd_wrapper)
    # denoising step
    xtilde = np.add(v,u)
    x = cnn_denoiser(xtilde, denoiser_model)
    # update u
    u = u+(v-x)
    imsave(os.path.join(output_dir,'pnp_output_itr_'+str(itr)+'.png'),x) 
    
    # calculate admm cost
    Gx = construct_forward_model(x,K,h,0)
    ggmrf_sum = 0
    for i in range(rows_hr):
      for j in range(cols_hr):
        for di in range(-1,2):
          for dj in range(-1,2):
            ggmrf_sum += g[di+1,dj+1] * abs(x[i,j]-x[(i+di)%rows_hr,(j+dj)%cols_hr])**p
    # divide by 2 because we counted each clique twice
    ggmrf_sum /= 2.
    cost_prior = ggmrf_sum/(p*sigx**p)
    admm_cost.append(sum(sum((Gx-y)**2))/(2*sigw*sigw) + cost_prior)
    residual_x.append(((x-x_old)**2).mean(axis=None)) 
    residual_v.append(((v-v_old)**2).mean(axis=None)) 
  # end ADMM recursive update
  plt.figure()
  plt.plot(list(range(admm_cost.__len__())), admm_cost)
  plt.xlabel('itr')
  plt.ylabel('admm cost')
  plt.savefig(os.path.join(output_dir,'admm_cost.png')) 
  plt.figure()
  plt.plot(list(range(residual_x.__len__())), residual_x, label="$\dfrac{1}{N}||x^{n+1}-x^n||^2$")
  plt.plot(list(range(residual_v.__len__())), residual_v, label="$\dfrac{1}{N}||v^{n+1}-v^n||^2$")
  plt.legend(loc='upper right')
  plt.xlabel('itr')
  plt.ylabel('residual')
  plt.savefig(os.path.join(output_dir,'residual.png')) 

  figout = 'pnp_output_'+filt_choice+'.png'
  imsave(os.path.join(output_dir,figout),x)
  return x


def optimization_wrapper(v,vtilde,y,h,K,itr,optim_method, pmap_model, icd_wrapper):
  if optim_method == 0:
    fvtilde = construct_forward_model(vtilde,K,h,0)
    H = pseudo_prox_map(np.subtract(y,fvtilde),pmap_model)
    v = np.add(vtilde, H)
  elif optim_method == 1:
    if itr == 0:
      for _ in range(10):
        v = icd_wrapper.update(v,vtilde)
    else:
      v = icd_wrapper.update(v,vtilde)
  else:
    raise Exception('Error: unknown optimization method.')
  return np.array(v)
