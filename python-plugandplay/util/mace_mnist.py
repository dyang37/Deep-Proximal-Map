import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import copy
from feed_model import cnn_denoiser, pseudo_prox_map_nonlinear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt
from proxMap import PM_rowproj, PM_colproj, PM_blockavg

def mace(z,y,yr,yc,yb,L,beta,mnist_model,dpm_model,denoiser_model,output_dir,rho=0.5,savefig=False):
  ################# Prepare output result dir
  
  # initialization  
  rows = 28
  cols = 28
  X = []
  W = []
  np.random.seed(2020)
  #X_img = np.random.rand(28,28)
  X_img = copy.deepcopy(z)
  output_dir = os.path.join(output_dir,"noiselessInit")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  for _ in range(beta.__len__()):
    X.append(copy.deepcopy(X_img))
    W.append(copy.deepcopy(X_img)) 
  # iterative reconstruction
  for itr in range(100):
    #X[0]: denoiser
    X[0] = np.reshape(cnn_denoiser(W[0], denoiser_model), (28,28))
    #X[1]: PM for row projector
    X[1] = PM_rowproj(W[1],yr,0.05,0.05)
    #X[2]: PM for col projector
    X[2] = PM_colproj(W[2],yc,0.05,0.05)
    #X[3]: PM for block averaging projector
    X[3] = PM_blockavg(W[3],yb,0.05,0.05,L=L)
    #X[4]: DPM
    AW_fm = np.reshape(mnist_model.predict(W[-1].reshape(1,28,28)), (10,))
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05])
    im = ax.imshow((y-AW_fm).reshape((1,10)), cmap='coolwarm',vmin=-0.1,vmax=0.1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'y-Ax_itr'+str(itr)+'.png')) 
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(W[-1], cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'v_itr'+str(itr)+'.png'))
    
    H = pseudo_prox_map_nonlinear(np.subtract(y,AW_fm),W[-1],dpm_model)
    X[-1] = np.clip(H+W[-1], 0, 1)
    Z = np.zeros((rows,cols))
    for i in range(beta.__len__()):
      Z += beta[i]*(2*X[i]-W[i])
    for i in range(beta.__len__()):
      W[i] += 2*rho*(Z-X[i])
    X_ret = np.zeros((rows,cols))
    for i in range(beta.__len__()):
      X_ret += beta[i]*X[i]
    err_img = X_ret - z
    Ax_ret = np.reshape(mnist_model.predict(X_ret.reshape(1,28,28)), (10,))
    y_err = y-Ax_ret 
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(H, cmap='coolwarm',vmin=-1,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'H_itr'+str(itr)+'.png'))
     
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(X_ret, cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_itr'+str(itr)+'.png'))
  # end ADMM recursive update
  rmse_x = sqrt(((X_ret-z)**2).mean(axis=None))
  norm2_y = ((y_err)**2).sum(axis=None)
  if savefig:    
    #with open(os.path.join(output_dir,'statistics.txt'), 'w') as filehandle:
    #  json.dump([beta,rmse_x,norm2_y.item()], filehandle)
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(z, cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_gd.png')) 
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(X_ret, cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_ret.png')) 
   
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05])
    im = ax.imshow(y_err.reshape((1,10)), cmap='coolwarm',vmin=-0.1,vmax=0.1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'err_y.png')) 
   
  return (rmse_x, norm2_y)

