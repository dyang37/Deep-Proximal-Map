import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "./denoisers/DnCNN"))
from skimage.io import imsave
from keras.models import  model_from_json
import copy
from dncnn import cnn_denoiser, pseudo_prox_map_nonlinear
from construct_forward_model import construct_nonlinear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def plug_and_play_nonlinear(y,sigma,alpha,sigw,gamma):
  # load pre-trained denoiser model
  output_dir = os.path.join(os.getcwd(),'./pnp_output/nonlinear/')
  denoiser_dir=os.path.join(os.getcwd(),'./denoisers/DnCNN')
  json_file = open(os.path.join(denoiser_dir,'model.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  denoiser_model = model_from_json(loaded_model_json)
  denoiser_model.load_weights(os.path.join(denoiser_dir,'model.h5'))
  
  # load pre-trained deep proximal map model
  pmap_dir=os.path.join(os.getcwd(),'cnn')
  pmap_model_name = "model_nonlinear_noiseless_hr"
  json_file = open(os.path.join(pmap_dir, pmap_model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  pmap_model = model_from_json(loaded_model_json)
  pmap_model.load_weights(os.path.join(pmap_dir,pmap_model_name+'.h5'))
  
  [rows, cols] = np.shape(y)
  x = np.random.rand(rows, cols)
  v = np.random.rand(rows, cols)
  u = np.zeros((rows, cols))
  # iterative reconstruction
  residual_x = []
  residual_v = []
  for itr in range(20):
    # forward step
    x_old = copy.deepcopy(x)
    v_old = copy.deepcopy(v)
    xtilde = np.subtract(v,u)
    fxtilde = construct_nonlinear_model(xtilde,sigma,alpha,0,gamma=gamma)
    H = pseudo_prox_map_nonlinear(np.subtract(y,fxtilde),xtilde,pmap_model)
    x = np.add(xtilde, H)
    # denoising step
    vtilde = np.add(x,u)
    v = cnn_denoiser(vtilde, denoiser_model)
    # update u
    u = u+(x-v)
    imsave(os.path.join(output_dir,'pnp_output_itr_'+str(itr)+'.png'),x) 
    residual_x.append(((x-x_old)**2).mean(axis=None)) 
    residual_v.append(((v-v_old)**2).mean(axis=None)) 
  # end ADMM recursive update
  plt.figure()
  plt.plot(list(range(residual_x.__len__())), residual_x, label="$\dfrac{1}{N}||x^{n+1}-x^n||^2$")
  plt.plot(list(range(residual_v.__len__())), residual_v, label="$\dfrac{1}{N}||v^{n+1}-v^n||^2$")
  plt.legend(loc='upper right')
  plt.xlabel('itr')
  plt.ylabel('residual')
  plt.savefig(os.path.join(output_dir,'residual.png')) 
  figout = 'pnp_output_nonlinear.png'
  imsave(os.path.join(output_dir,figout),v)
  return x


