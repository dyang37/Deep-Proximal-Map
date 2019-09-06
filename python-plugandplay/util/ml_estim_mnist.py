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
from feed_model import pseudo_prox_map_nonlinear
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import model_from_json
import copy

def ml_estimate_mnist(y,sig,sigw,d,_log_data):
  perterb = 1e-30
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  rows = 28
  cols = 28
  # read DPM model for mnist classifier
  dpm_model_dir=os.path.join(os.getcwd(),'../cnn/dpm_model_mnist/mnist_mixed/')
  if _log_data:
    dpm_model_name = "model_mixed3_log_mnist_sig_"+str(sig)+"_sigw"+str(sigw)
  else:
    dpm_model_name = "model_mixed3_mnist_sig_"+str(sig)+"_sigw"+str(sigw)
  json_file = open(os.path.join(dpm_model_dir, dpm_model_name+'.json'), 'r')
  dpm_model_json = json_file.read()
  json_file.close()
  dpm_model = model_from_json(dpm_model_json)
  # load weights into new model
  dpm_model.load_weights(os.path.join(dpm_model_dir,dpm_model_name+'.h5'))
  print("Loaded DPM model from disk")
  
  # read mnist classifier model
  mnist_model_dir = os.path.join(os.getcwd(),'../cnn')
  mnist_model_name = "mnist_forward_newModel"
  json_file = open(os.path.join(mnist_model_dir, mnist_model_name+'.json'), 'r')
  mnist_model_json = json_file.read()
  json_file.close()
  mnist_model = model_from_json(mnist_model_json)
  # load weights into model
  mnist_model.load_weights(os.path.join(mnist_model_dir, mnist_model_name+".h5"))
  print("Loaded mnist model from disk")
  
  
  # iterative reconstruction
  idx = -20
  d = y_train[idx]
  x = x_train[idx]/255.+ np.random.normal(0,0.3,(rows,cols))
  #x = x_train[idx+1]/255.
  y = np.reshape(mnist_model.predict(x_train[idx].reshape((1,rows,cols))), (10,))
  #x = np.zeros((rows,cols))
  np.random.seed(2019)
  #x = np.random.rand(rows,cols)
  fig, ax = plt.subplots()
  cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
  im = ax.imshow(x, cmap='coolwarm',vmin=x.min(),vmax=x.max())
  fig.colorbar(im, cax=cax, orientation='horizontal')
  plt.savefig('./x_gd.png')
  plt.close()
  print("init digit: ",d)
  output_dir = os.path.join(os.getcwd(),'../results/ml_output_mnist/mnist_mixed/')
  if _log_data:
    output_dir = os.path.join(output_dir,'log')
  else:
    output_dir = os.path.join(output_dir,'nolog')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  print("output stored in ", output_dir)
  imsave(os.path.join(output_dir,'v_init.png'), np.clip(x,0,None))
  ml_cost = []
  for itr in range(40):
    print('iteration ',itr)
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(x, cmap='coolwarm',vmin=x.min(),vmax=x.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'x_itr'+str(itr)+'.png'))

    Ax = np.reshape(mnist_model.predict(x.reshape((1,rows,cols))), (10,))
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(Ax.reshape((1,10)), cmap='coolwarm',vmin=Ax.min(),vmax=Ax.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'Ax_itr'+str(itr)+'.png'))
    plt.close()

    ml_cost.append(sqrt(((y-Ax)**2).mean(axis=None)))
    err_y = np.reshape(y-Ax, (1,10))
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_y, cmap='coolwarm',vmin=-err_y.max(),vmax=err_y.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'y_err_pml_itr'+str(itr)+'.png'))
    
    # deep proximal map update
    if _log_data:
      H = pseudo_prox_map_nonlinear(np.subtract(np.log10(y+perterb),np.log10(Ax+perterb)),x,dpm_model)
    else:
      H = pseudo_prox_map_nonlinear(np.subtract(y,Ax),x,dpm_model)
    x = np.clip(np.add(x, H),0,1)

  y_cnn = np.reshape(mnist_model.predict(x.reshape((1,rows,cols))), (10,))
  print('prediction of generated image:',y_cnn)
  mse_y_gd = ((y-y_cnn)**2).mean(axis=None)
  print('pixelwise mse value for y between cnn and groundtruth: ',mse_y_gd)
  
  # cost function plot
  plt.figure()
  plt.plot(list(range(ml_cost.__len__())),ml_cost,label="PML with deep prox map")
  plt.legend(loc='upper left')
  plt.xlabel('iteration')
  plt.ylabel('ML cost $\sqrt{\dfrac{1}{N}||Y-A(x)||^2}$')
  plt.savefig(os.path.join(output_dir,'ml_cost.png'))

  # save output images
  imsave(os.path.join(output_dir,'ml_output_cnn.png'), np.clip(x,0,1))
  print("Done.")
  return

