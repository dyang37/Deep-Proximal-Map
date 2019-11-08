import numpy as np
import sys
import os
from math import sqrt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from feed_model import pseudo_prox_map_nonlinear, pseudo_prox_map
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.models import model_from_json
import copy
import pickle

def ml_estimate_mnist(y,sig,sigw,d):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  x_train = x_train.astype('float64')
  x_train /= 255.
  rows = 28
  cols = 28
  # read DPM model for mnist classifier
  dpm_model_dir=os.path.join(os.getcwd(),'../cnn/dpm_model_mnist/mnist_mixed/')
  dpm_model_name = "model_dense_mixed4_mnist_dense"
  print("DPM model used:",dpm_model_name)
  json_file = open(os.path.join(dpm_model_dir, dpm_model_name+'.json'), 'r')
  dpm_model_json = json_file.read()
  json_file.close()
  dpm_model = model_from_json(dpm_model_json)
  # load weights into new model
  dpm_model.load_weights(os.path.join(dpm_model_dir,dpm_model_name+'.h5'))
  print("Loaded DPM model from disk")
  
  # read mnist classifier model
  mnist_model_dir = os.path.join(os.getcwd(),'../cnn')
  mnist_model_name = "mnist_forward_autoencoder"
  json_file = open(os.path.join(mnist_model_dir, mnist_model_name+'.json'), 'r')
  mnist_model_json = json_file.read()
  json_file.close()
  mnist_model = model_from_json(mnist_model_json)
  # load weights into model
  mnist_model.load_weights(os.path.join(mnist_model_dir, mnist_model_name+".h5"))
  print("Loaded mnist model from disk")
  
  
  np.random.seed(2017)
  # iterative reconstruction
  idx = -20
  d = y_train[idx]
  #x = x_train[idx]+ np.random.normal(0,0.1,(rows*cols,))
  y = np.reshape(mnist_model.predict(x_train[idx].reshape((1,28,28))), (10,))
  #x = np.zeros((rows,cols))
  x = np.random.rand(rows,cols)
  fig, ax = plt.subplots()
  cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
  im = ax.imshow(x, cmap='coolwarm',vmin=0,vmax=1)
  fig.colorbar(im, cax=cax, orientation='horizontal')
  plt.savefig('./x_gd.png')
  plt.close()
  print("init digit: ",d)
  output_dir = os.path.join(os.getcwd(),'../results/ml_output_mnist/mnist_mixed/')
  output_dir = os.path.join(output_dir,'dense')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  print("output stored in ", output_dir)
  imsave(os.path.join(output_dir,'v_init.png'), np.clip(x,0,None))
  ml_cost = []
  err_y_list = []
  for itr in range(100):
    print('iteration ',itr)
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(x, cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_itr'+str(itr)+'.png'))

    Ax = np.reshape(mnist_model.predict(x.reshape((1,rows,cols))), (10,))
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(Ax.reshape((1,10)), cmap='coolwarm',vmin=0,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'Ax_itr'+str(itr)+'.png'))
    plt.close()

    ml_cost.append(sqrt(((y-Ax)**2).sum()))
    err_y = np.reshape(y-Ax, (1,10))
    err_y_list.append(err_y)
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.10, 0.5, 0.05])
    im = ax.imshow(err_y, cmap='coolwarm',vmin=-0.01,vmax=0.01)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'y_err_pml_itr'+str(itr)+'.png'))
    # deep proximal map update
    H = pseudo_prox_map_nonlinear(np.subtract(y,Ax),x,dpm_model)
    x = np.clip(np.add(x, H),0,1)

  y_cnn = np.reshape(mnist_model.predict(x.reshape((1,rows,cols))), (10,))
  print('prediction of generated image:',y_cnn)
  mse_y_gd = ((y-y_cnn)**2).mean(axis=None)
  print('pixelwise mse value for y between cnn and groundtruth: ',mse_y_gd)
  
  # cost function plot
  plt.figure()
  plt.semilogy(list(range(ml_cost.__len__())),ml_cost,label="PML with deep prox map")
  plt.legend(loc='upper left')
  plt.xlabel('iteration')
  plt.ylabel('ML cost $\sqrt{||Y-A(x)||^2}$')
  plt.savefig(os.path.join(output_dir,'ml_cost.png'))
  plt.figure()
  plt.semilogy(list(range(20)),ml_cost[:20],label="PML with deep prox map")
  plt.legend(loc='upper left')
  plt.xlabel('iteration')
  plt.ylabel('ML cost $\sqrt{||Y-A(x)||^2}$')
  plt.savefig(os.path.join(output_dir,'ml_cost_20.png'))

  # save experiment data
  exp_data = {"ml_cost":ml_cost, "error":err_y_list}
  fd = open(os.path.join(output_dir,"exp_data.dat"),'wb')
  pickle.dump(exp_data, fd) 
  print("Done.")
  return

