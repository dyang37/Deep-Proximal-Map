import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imsave
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.models import  model_from_json
import copy
from dncnn import cnn_denoiser, pseudo_prox_map_nonlinear
from gmrf import gmrf_denoiser
from construct_forward_model import construct_nonlinear_model
from grad import grad_nonlinear_tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from skimage.restoration import denoise_tv_chambolle as denoiser_tv
from math import sqrt

def mace(p,z,y,sigma_g,alpha,beta,sigw,sig,gamma,clip,rho=0.5):
  # load pre-trained denoiser model
  print("rho=",rho)
  if p == 1:
    print("using tv as prior")
    output_dir = os.path.join(os.getcwd(),'../results/mace/nonlinear/tv')
  elif p == 2:
    print("using gmrf as prior")
    output_dir = os.path.join(os.getcwd(),'../results/mace/nonlinear/gmrf')
  else:
    print("using dncnn as prior")
    output_dir = os.path.join(os.getcwd(),'../results/mace/nonlinear/dncnn')
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
  # approximate sig_x using original image z
  g = np.array([[1./12., 1./6., 1./12.],[1./6.,0,1./6.],[1./12.,1./6.,1./12.]])
  sigx = 0
  for i in range(rows):
    for j in range(cols):
      for di in range(-1,2):
        for dj in range(-1,2):
          sigx += g[di+1,dj+1] * abs((z[i,j]-z[(i+di)%rows,(j+dj)%cols]))**p
            # divide by 2 because we counted each clique twice
  sigx /= 2.
  sigx = (sigx/(rows*cols))**(1./p)
  print('estimated GGMRF sigma = ',sigx) 
  # initialization  
  X1 = np.random.rand(rows, cols)
  X2 = copy.deepcopy(X1)
  W1 = copy.deepcopy(X1)
  W2 = copy.deepcopy(X1)
  # iterative reconstruction
  for itr in range(50):
    print("iteration ",itr)
    sig_n = sqrt(beta)*sig
    if p==1:
      X1 = denoiser_tv(W1, sig_n)
    elif p==2:
      X1 = gmrf_denoiser(X1,W1,sig_n,sigx)
    else:
      X1 = cnn_denoiser(W1, denoiser_model)
    
    # calculate admm cost
    AW2 = construct_nonlinear_model(W2,sigma_g,alpha,0,gamma=gamma,clip=clip)
    X2 = pseudo_prox_map_nonlinear(np.subtract(y,AW2),W2,pmap_model) + W2
    Z = X1+X2-0.5*(W1+W2)
    W1 = W1 + 2*rho*(Z-X1)
    W2 = W2 + 2*rho*(Z-X2)
    imsave(os.path.join(output_dir,'grad_output_itr_'+str(itr)+'.png'),np.clip(X1,0,1)) 

  X_ret = 0.5*(X1+X2)
  imsave(os.path.join(output_dir,'mace_output.png'),np.clip(X_ret,0,1)) 
  # end ADMM recursive update
  return X_ret


