import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
from keras.models import  model_from_json
import copy
from feed_model import cnn_denoiser, pseudo_prox_map_nonlinear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt
from proj import rowproj, colproj

def mace(z,y,yr,yc,beta,mnist_model,dpm_model,denoiser_model,output_dir,rho=0.5,savefig=False):
  ################# Prepare output result dir
  if savefig:
    with open(os.path.join(output_dir,'beta.txt'), 'w') as filehandle:
        json.dump(beta, filehandle)
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(z, cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_gd.png')) 
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(yr.reshape((1,28)), cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'yr.png')) 
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(yc.reshape((1,28)), cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'yc.png')) 

  # initialization  
  rows = 28
  cols = 28
  X = []
  W = []
  np.random.seed(2020)
  X_img = np.random.rand(28,28)
  #X_img = copy.deepcopy(z)
  if savefig:
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(X_img, cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_init.png')) 

  for _ in range(beta.__len__()):
    X.append(copy.deepcopy(X_img))
    W.append(copy.deepcopy(X_img)) 
  # iterative reconstruction
  for itr in range(100):
    #X[0]: denoiser
    X[0] = np.reshape(cnn_denoiser(W[0], denoiser_model), (28,28))
    #X[1]: row projector
    X[1] = rowproj(W[1],yr,0.05,0.05)
    #X[2]: col projector
    X[2] = colproj(W[2],yc,0.05,0.05)
    #X[3]: DPM
    AW_fm = np.reshape(mnist_model.predict(W[-1].reshape((1,rows*cols))), (10,))
    X[-1] = np.clip(np.reshape(pseudo_prox_map_nonlinear(np.subtract(y,AW_fm),W[-1].reshape((28*28,)),dpm_model),(28,28)) + W[-1],0,1)
    Z = np.zeros((rows,cols))
    for i in range(beta.__len__()):
      Z += beta[i]*(2*X[i]-W[i])
    for i in range(beta.__len__()):
      W[i] += 2*rho*(Z-X[i])
    X_ret = np.zeros((rows,cols))
    for i in range(beta.__len__()):
      X_ret += beta[i]*X[i]
    err_img = X_ret - z
    # end ADMM recursive update
    Ax_ret = np.reshape(mnist_model.predict(X_ret.reshape((1,rows*cols))), (10,))
    if savefig:    
      fig, ax = plt.subplots()
      cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
      im = ax.imshow(X_ret.reshape((28,28)), cmap='coolwarm',vmin=0,vmax=1)
      fig.colorbar(im, cax=cax, orientation='horizontal')
      plt.savefig(os.path.join(output_dir,'x_itr'+str(itr)+'.png')) 
     
      y_err = y-Ax_ret 
      fig, ax = plt.subplots()
      cax = fig.add_axes([0.27, 0.1, 0.5, 0.05])
      im = ax.imshow(y_err.reshape((1,10)), cmap='coolwarm',vmin=-abs(y_err).max(),vmax=abs(y_err).max())
      fig.colorbar(im, cax=cax, orientation='horizontal')
      plt.savefig(os.path.join(output_dir,'err_y_itr'+str(itr)+'.png')) 

  return sqrt(((X_ret-z)**2).mean(axis=None))

