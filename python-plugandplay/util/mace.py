import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from DnCNN import DnCNN
from skimage.io import imsave
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.models import  model_from_json
import copy
from feed_model import cnn_denoiser, pseudo_prox_map_nonlinear
from construct_forward_model import construct_nonlinear_model
from grad import grad_nonlinear_tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from math import sqrt

def mace2(z,y,sigma_g,alpha,beta,sigw,nl,sig,gamma,clip,w=0.5,rho=0.5, savefig=False):
  # load pre-trained denoiser model
  if savefig:
    output_dir = os.path.join(os.getcwd(),'../results/mace2')
    output_dir = os.path.join(output_dir,"sigma_"+str(nl))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  denoiser_name = "model_109"
  denoiser_dir="/home/yang1467/deepProxMap/my_plug_and_play/python-plugandplay/denoisers/DnCNN/DnCNN/TrainingCodes/dncnn_keras/models/"
  denoiser_dir=os.path.join(denoiser_dir,'DnCNN_sigma'+str(nl))
  denoiser_name = "model_109"
  denoiser_model = DnCNN(depth=17,filters=64,image_channels=1,use_bnorm=True)
  denoiser_model.load_weights(os.path.join(denoiser_dir,denoiser_name+'.hdf5'))
  
  # load pre-trained deep proximal map model
  pmap_dir=os.path.join(os.getcwd(),'../cnn/nonlinear_model')
  pmap_model_name = "model_nonlinear_"
  if clip:
    pmap_model_name += "clip_sigw_"
  else:
    pmap_model_name += "noclip_sigw_"
  pmap_model_name += str(sigw) 
  json_file = open(os.path.join(pmap_dir, pmap_model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  pmap_model = model_from_json(loaded_model_json)
  pmap_model.load_weights(os.path.join(pmap_dir,pmap_model_name+'.h5'))
  [rows, cols] = np.shape(y)
  # initialization  
  X1 = np.random.rand(rows, cols)
  X2 = copy.deepcopy(X1)
  W1 = copy.deepcopy(X1)
  W2 = copy.deepcopy(X1)
  # iterative reconstruction
  for itr in range(20):
    sig_n = sqrt(beta)*sig
    X1 = cnn_denoiser(W1, denoiser_model)
    
    # calculate admm cost
    AW2 = construct_nonlinear_model(W2,sigma_g,alpha,0,gamma=gamma,clip=clip)
    X2 = pseudo_prox_map_nonlinear(np.subtract(y,AW2),W2,pmap_model) + W2
    Z = w*(2*X1-W1) + (1.-w)*(2*X2-W2)
    W1 = W1 + 2*rho*(Z-X1)
    W2 = W2 + 2*rho*(Z-X2)
    X_ret = w*X1+(1-w)*X2
    err_img = X_ret - z
    # end ADMM recursive update
  if savefig:
    imsave(os.path.join(output_dir,'mace_output_w_'+str(w)+'.png'),np.clip(X_ret,0,1))
  return sqrt(((X_ret-z)**2).mean(axis=None))

def mace3(z,y,sigma_g,alpha,beta,sigw,nl_list,sig,gamma,clip,w=0.5,rho=0.5, savefig=False):
  # load pre-trained denoiser model
  if savefig:
    output_dir = os.path.join(os.getcwd(),'../results/mace3/')
    output_dir = os.path.join(output_dir,'optimal/')
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  denoiser_name = "model_100"
  dncnn_model_list = []
  for nl in nl_list:
    denoiser_dir="/home/yang1467/deepProxMap/my_plug_and_play/python-plugandplay/denoisers/DnCNN/DnCNN/TrainingCodes/dncnn_keras/models/"
    denoiser_dir=os.path.join(denoiser_dir,'DnCNN_sigma'+str(nl))
    denoiser_name = "model_100"
    denoiser_model = DnCNN(depth=17,filters=64,image_channels=1,use_bnorm=True)
    denoiser_model.load_weights(os.path.join(denoiser_dir,denoiser_name+'.hdf5'))
    dncnn_model_list.append(denoiser_model)
  # load pre-trained deep proximal map model
  pmap_dir=os.path.join(os.getcwd(),'../cnn/nonlinear_model')
  pmap_model_name = "model_nonlinear_"
  if clip:
    pmap_model_name += "clip_sigw_"
  else:
    pmap_model_name += "noclip_sigw_"
  pmap_model_name += str(sigw) 
  json_file = open(os.path.join(pmap_dir, pmap_model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  pmap_model = model_from_json(loaded_model_json)
  pmap_model.load_weights(os.path.join(pmap_dir,pmap_model_name+'.h5'))
  [rows, cols] = np.shape(y)
  # initialization  
  X1 = np.random.rand(rows, cols)
  X2 = copy.deepcopy(X1)
  X3 = copy.deepcopy(X1)
  W1 = copy.deepcopy(X1)
  W2 = copy.deepcopy(X1)
  W3 = copy.deepcopy(X1)
  
  # iterative reconstruction
  for itr in range(20):
    sig_n = sqrt(beta)*sig
    X1 = cnn_denoiser(W1, dncnn_model_list[0])
    X2 = cnn_denoiser(W2, dncnn_model_list[1])
    
    AW3 = construct_nonlinear_model(W3,sigma_g,alpha,0,gamma=gamma,clip=clip)
    X3 = pseudo_prox_map_nonlinear(np.subtract(y,AW3),W3,pmap_model) + W3
    Z = w*((2*X1-W1)+(2*X2-W2))/2. + (1.-w)*(2*X3-W3)
    W1 = W1 + 2*rho*(Z-X1)
    W2 = W2 + 2*rho*(Z-X2)
    W3 = W2 + 2*rho*(Z-X3)
    X_ret = w*(X1+X2)/2+(1-w)*X3
    err_img = X_ret - z
    # end ADMM recursive update
  if savefig:
    imsave(os.path.join(output_dir,'mace_output_w_'+str(w)+'.png'),np.clip(X_ret,0,1))
  return sqrt(((X_ret-z)**2).mean(axis=None))

