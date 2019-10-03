import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.models import  model_from_json
import copy
from feed_model import cnn_denoiser, pseudo_prox_map_nonlinear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt
from icdproj_wrapper import icdproj_wrapper
def mace(z,y,yr,yc,beta,mnist_model,dpm_model,denoiser_model,rho=0.5, savefig=False):
  ################# Prepare output result dir
  if savefig:
    output_dir = os.path.join(os.getcwd(),'../results/mace_mnist/')
    output_dir = os.path.join(output_dir,str(sum(beta[:-1])))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  # initialization  
  rows = 28
  cols = 28
  X = []
  W = []
  np.random.seed(2020)
  X_img = np.random.rand(28,28)
  for _ in range(3):
    X.append(copy.deepcopy(X_img))
    W.append(copy.deepcopy(X_img)) 
  # iterative reconstruction
  for itr in range(30):
    #X[0]: denoiser
    X[0] = np.reshape(cnn_denoiser(W[0], denoiser_model), (28,28))
    #X[1]: back projector
    X[1] = icdproj_wrapper(yr,yc,W[1],X[1],1.,itr)
    #X[2]: DPM
    AW_fm = np.reshape(mnist_model.predict(W[2].reshape((1,rows*cols))), (10,))
    X[2] = np.clip(np.reshape(pseudo_prox_map_nonlinear(np.subtract(y,AW_fm),W[2].reshape((28*28,)),dpm_model),(28,28)) + W[2],0,1)
    Z = beta[0]*(2*X[0]-W[0])+beta[1]*(2*X[1]-W[1])+beta[2]*(2*X[2]-W[2])
    for i in range(3):
      W[i] = W[i] + 2*rho*(Z-X[i])
    X_ret = beta[0]*X[0] + beta[1]*X[1] + beta[2]*X[2]
    err_img = X_ret - z
    # end ADMM recursive update
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(X_ret.reshape((28,28)), cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_itr'+str(itr)+'.png')) 
   
    Ax_ret = np.reshape(mnist_model.predict(X_ret.reshape((1,rows*cols))), (10,))
    y_err = y-Ax_ret 
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05])
    im = ax.imshow(y_err.reshape((1,10)), cmap='coolwarm',vmin=-abs(y_err).max(),vmax=abs(y_err).max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'err_y_itr'+str(itr)+'.png')) 

  return sqrt(((X_ret-z)**2).mean(axis=None))

