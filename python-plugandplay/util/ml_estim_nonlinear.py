import numpy as np
import sys
import os
from math import sqrt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imsave
from scipy.misc import imresize
from forward_model import camera_model
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

def ml_estimate_nonlinear(y,sigma_g,alpha,sig,sigw,gamma,clip):
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
  x = np.random.rand(rows,cols)
  print(output_dir)
  imsave(os.path.join(output_dir,'v_init.png'), np.clip(x,0,None))
  ml_cost = []
  grad_diff_mse = []
  for itr in range(50):
    print('iteration ',itr)
    fx = camera_model(x,sigma_g,alpha,0,gamma=gamma, clip=clip)
    ml_cost.append(sqrt(((y-fx)**2).mean(axis=None)))
    imsave(os.path.join(output_dir,'pml_output_itr'+str(itr)+'.png'), np.clip(x,0,1))
    err_y = y-fx
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_y, cmap='coolwarm',vmin=-err_y.max(),vmax=err_y.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'y_err_pml_itr'+str(itr+1)+'.png'))
    
    # deep proximal map update
    grad_f_tf = grad_nonlinear_tf(x,y,sigma_g,alpha,sigw,gamma,clip=clip)
    grad_f = grad_nonlinear(x,y,sigma_g,alpha,sigw,gamma,clip=clip)
    sig_gradf_tf = -sig*sig*grad_f_tf
    sig_gradf = -sig*sig*grad_f
    H = pseudo_prox_map_nonlinear(np.subtract(y,fx),x,model)
    grad_diff_mse.append(sqrt(((sig_gradf-H)**2).mean(axis=None)))

    x = np.clip(np.add(x, H),0,None)
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
 
  y_cnn = camera_model(x,sigma_g,alpha,0,gamma=gamma, clip=clip)
  mse_y_gd = ((y-y_cnn)**2).mean(axis=None)
  print('pixelwise mse value for y between cnn and groundtruth: ',mse_y_gd)
  
  # cost function plot
  plt.figure()
  plt.semilogy(list(range(grad_diff_mse.__len__())),grad_diff_mse)
  plt.legend(loc='upper left')
  plt.xlabel('iteration')
  plt.ylabel('mse')
  plt.savefig(os.path.join(output_dir,'grad_diff_mse.png'))
  plt.figure()
  plt.semilogy(list(range(ml_cost.__len__())),ml_cost,label="PML with deep prox map")
  plt.ylim(0.01, 0.4)
  plt.legend(loc='upper left')
  plt.xlabel('iteration')
  plt.ylabel('ML cost $\sqrt{\dfrac{1}{N}||Y-A(x)||^2}$')
  plt.savefig(os.path.join(output_dir,'ml_cost.png'))

  # save output images
  imsave(os.path.join(output_dir,'y_input.png'), np.clip(y,0,1))
  imsave(os.path.join(output_dir,'ml_output_cnn.png'), np.clip(x,0,1))
  imsave(os.path.join(output_dir,'forward_modeled_cnn.png'), np.clip(y_cnn,0,1))
  print("Done.")
  return

