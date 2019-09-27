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

def mace(z,y,sigw,sig,w,mnist_model,dpm_model,dncnn_model_list,rho=0.5, savefig=False):
  ################# Prepare output result dir
  if savefig:
    output_dir = os.path.join(os.getcwd(),'../results/mace_mnist/')
    output_dir = os.path.join(output_dir,'optimal/')
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  # initialization  
  rows = 28
  cols = 28
  X = []
  W = []
  X_img = np.random.rand(28*28,)
  for _ in range(dncnn_model_list.__len__()+1):
    X.append(copy.deepcopy(X_img))
    W.append(copy.deepcopy(X_img)) 
  # iterative reconstruction
  for itr in range(20):
    for i in range(dncnn_model_list.__len__()):
      #X[i] = np.reshape(cnn_denoiser(W[i].reshape(28,28), dncnn_model_list[i]), (28*28,))
      X[i] = W[i]
    AW_fm = np.reshape(mnist_model.predict(W[-1].reshape((1,rows*cols))), (10,))
    X[-1] = np.clip(pseudo_prox_map_nonlinear(np.subtract(y,AW_fm),W[-1],dpm_model) + W[-1],0,1)
    Z = np.zeros((rows*cols,))
    for i in range(dncnn_model_list.__len__()):
      Z += w[i]*(2*X[i]-W[i])
    Z += (1-sum(w))* (2*X[-1]-W[-1])
    for i in range(dncnn_model_list.__len__()+1):
      W[i] = W[i] + 2*rho*(Z-X[i])
    X_ret = np.zeros((rows*cols))
    for i in range(dncnn_model_list.__len__()):
      X_ret += w[i]*X[i]
    X_ret += (1-sum(w))*X[-1]
    err_img = X_ret - z
    # end ADMM recursive update
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(X_ret.reshape((28,28)), cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_itr'+str(itr)+'.png')) 
  return sqrt(((X_ret-z)**2).mean(axis=None))

