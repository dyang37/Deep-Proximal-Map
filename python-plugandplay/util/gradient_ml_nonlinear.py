import numpy as np
import sys
import os
from math import sqrt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imsave
from scipy.misc import imresize
from construct_forward_model import construct_nonlinear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from dncnn import pseudo_prox_map_nonlinear
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import model_from_json
from grad import grad_nonlinear, grad_nonlinear_tf
import copy

def ml_gradient_nonlinear(y,sigma_g,alpha,sig,sigw,gamma,clip):
  output_dir = os.path.join(os.getcwd(),'../results/ml_output_nonlinear/')
  [rows, cols] = np.shape(y)
  N = rows*cols
  # initialize cpp wrapper for icd
  # read pre-trained model for pseudo-proximal map
  model_dir=os.path.join(os.getcwd(),'../cnn')
  if clip:
    model_name = "model_nonlinear_noisy_clip"
    output_dir = os.path.join(output_dir,'clip')
  else:
    model_name = "model_nonlinear_noisy_noclip"
    output_dir = os.path.join(output_dir,'noclip')
  print("deep pmap model: ",model_name)
  if sigw == 0:
    output_dir = os.path.join(output_dir,'noiseless')
  else:
    output_dir = os.path.join(output_dir,'noisy')
  print("output stored in ", output_dir)
  json_file = open(os.path.join(model_dir, model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights(os.path.join(model_dir,model_name+'.h5'))
  # iterative reconstruction
  x_pml = np.random.rand(rows,cols)
  x_grad = copy.deepcopy(x_pml) 
  print(output_dir)
  imsave(os.path.join(output_dir,'v_init.png'), np.clip(x_pml,0,None))
  ml_cost_pml = []
  ml_cost_grad = []
  for itr in range(50):
    print('iteration ',itr)
    fx_pml = construct_nonlinear_model(x_pml,sigma_g,alpha,0,gamma=gamma, clip=clip)
    fx_grad = construct_nonlinear_model(x_grad,sigma_g,alpha,0,gamma=gamma, clip=clip)
    ml_cost_pml.append(sqrt(((y-fx_pml)**2).mean(axis=None)))
    ml_cost_grad.append(sqrt(((y-fx_grad)**2).mean(axis=None)))
    imsave(os.path.join(output_dir,'pml_output_itr'+str(itr+1)+'.png'), np.clip(x_pml,0,1))
    imsave(os.path.join(output_dir,'gradml_output_itr'+str(itr+1)+'.png'), np.clip(x_grad,0,1))
    err_y_pml = y-fx_pml
    err_y_grad = y-fx_grad
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_y_pml, cmap='coolwarm',vmin=-err_y_pml.max(),vmax=err_y_pml.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'y_err_pml_itr'+str(itr+1)+'.png'))
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_y_grad, cmap='coolwarm',vmin=-err_y_pml.max(),vmax=err_y_pml.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'y_err_grad_itr'+str(itr+1)+'.png'))

    # deep proximal map update
    grad_f_tf = grad_nonlinear_tf(x_grad,y,sigma_g,alpha,sigw,gamma,clip=clip)
    grad_f = grad_nonlinear(x_grad,y,sigma_g,alpha,sigw,gamma,clip=clip)
    sig_gradf_tf = -sig*sig*grad_f_tf
    sig_gradf = -sig*sig*grad_f
    H = pseudo_prox_map_nonlinear(np.subtract(y,fx_pml),x_pml,model)
    x_pml = np.clip(np.add(x_pml, H),0,None)
    x_grad = np.clip(np.add(x_grad, sig_gradf_tf),0,1)
    # make plots
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(H, cmap='coolwarm',vmin=-H.max(),vmax=H.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'H_itr'+str(itr+1)+'.png'))
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(sig_gradf_tf, cmap='coolwarm',vmin=-H.max(),vmax=H.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'gradient_tf_itr'+str(itr+1)+'.png'))
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(sig_gradf, cmap='coolwarm',vmin=-H.max(),vmax=H.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'gradient_itr'+str(itr+1)+'.png'))
 
  y_cnn = construct_nonlinear_model(x_pml,sigma_g,alpha,0,gamma=gamma, clip=clip)
  mse_y_gd = ((y-y_cnn)**2).mean(axis=None)
  print('pixelwise mse value for y between cnn and groundtruth: ',mse_y_gd)
  
  # cost function plot
  plt.figure()
  plt.semilogy(list(range(ml_cost_pml.__len__())),ml_cost_pml,label="PML with deep prox map")
  plt.semilogy(list(range(ml_cost_grad.__len__())),ml_cost_grad,label="gradient ML reconstruction")
  plt.ylim(0.01, 0.4)
  plt.legend(loc='upper left')
  plt.xlabel('iteration')
  plt.ylabel('ML cost $\sqrt{\dfrac{1}{N}||Y-A(x)||^2}$')
  plt.savefig(os.path.join(output_dir,'ml_cost.png'))
  # save output images
  imsave(os.path.join(output_dir,'y_input.png'), np.clip(y,0,1))
  imsave(os.path.join(output_dir,'ml_output_cnn.png'), np.clip(x_pml,0,1))
  imsave(os.path.join(output_dir,'forward_modeled_cnn.png'), np.clip(y_cnn,0,1))
  print("Done.")
  return

