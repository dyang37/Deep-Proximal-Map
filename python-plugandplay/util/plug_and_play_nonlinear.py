import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imsave
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import  model_from_json
import copy
from dncnn import cnn_denoiser, pseudo_prox_map_nonlinear
from construct_forward_model import construct_nonlinear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plug_and_play_nonlinear(z,y,sigma,alpha,sigw,gamma,clip):
  # load pre-trained denoiser model
  output_dir = os.path.join(os.getcwd(),'../results/pnp_output/nonlinear/')
  imsave(os.path.join(output_dir,'y_input.png'), np.clip(y,0,1))
  denoiser_dir=os.path.join(os.getcwd(),'../denoisers/DnCNN')
  json_file = open(os.path.join(denoiser_dir,'model.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  denoiser_model = model_from_json(loaded_model_json)
  denoiser_model.load_weights(os.path.join(denoiser_dir,'model.h5'))
  
  # load pre-trained deep proximal map model
  pmap_dir=os.path.join(os.getcwd(),'../cnn')
  pmap_model_name = "model_nonlinear_noisy_"
  if clip:
    pmap_model_name += "clip"
  else:
    pmap_model_name += "noclip"
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
  for itr in range(30):
    # forward step
    x_old = copy.deepcopy(x)
    v_old = copy.deepcopy(v)
    xtilde = np.subtract(v,u)
    fxtilde = construct_nonlinear_model(xtilde,sigma,alpha,0,gamma=gamma, clip=clip)
    H = pseudo_prox_map_nonlinear(np.subtract(y,fxtilde),xtilde,pmap_model)
    x = np.clip(np.add(xtilde, H), 0, None)
    # denoising step
    vtilde = np.add(x,u)
    v = cnn_denoiser(vtilde, denoiser_model)
    # update u
    u = u+(x-v)
    imsave(os.path.join(output_dir,'pnp_output_itr_'+str(itr)+'.png'),x) 
    residual_x.append(((x-x_old)**2).mean(axis=None)) 
    residual_v.append(((v-v_old)**2).mean(axis=None)) 
    
    err_img = x-z
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_img, cmap='coolwarm',vmin=-err_img.max(),vmax=err_img.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'err_itr'+str(itr+1)+'.png'))
  # end ADMM recursive update
  plt.figure()
  plt.semilogy(list(range(residual_x.__len__())), residual_x, label="$log\{\dfrac{1}{N}||x^{n+1}-x^n||^2\}$")
  plt.semilogy(list(range(residual_v.__len__())), residual_v, label="$log\{\dfrac{1}{N}||v^{n+1}-v^n||^2\}$")
  plt.legend(loc='upper right')
  plt.xlabel('itr')
  plt.ylabel('residual')
  plt.savefig(os.path.join(output_dir,'residual.png')) 
  figout = 'pnp_output_nonlinear.png'
  imsave(os.path.join(output_dir,figout),v)
  return x


