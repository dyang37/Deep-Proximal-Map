import os,sys
import numpy as np
import json
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from feed_model import pseudo_prox_map_nonlinear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import model_from_json
import tensorflow as tf
import copy

outdir_name = "MACE5_SoA"
# load mnist model
mnist_model_dir = os.path.join(os.getcwd(),'../cnn')
mnist_model_name = "mnist_forward_cnn"
json_file = open(os.path.join(mnist_model_dir, mnist_model_name+'.json'), 'r')
mnist_model_json = json_file.read()
json_file.close()
mnist_model = model_from_json(mnist_model_json)
mnist_model.load_weights(os.path.join(mnist_model_dir, mnist_model_name+".h5"))
# load Deep proximal map model
dpm_model_dir=os.path.join(os.getcwd(),'../cnn/dpm_model_mnist/mnist_mixed/')
dpm_model_name = "model_cnn_mixed4_mnist_cnn_laplace0.05"
json_file = open(os.path.join(dpm_model_dir, dpm_model_name+'.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
dpm_model = model_from_json(loaded_model_json)
dpm_model.load_weights(os.path.join(dpm_model_dir,dpm_model_name+'.h5'))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float64')
x_train /= 255.

exp_dir = os.path.join(os.getcwd(),'../results/mace_mnist_cnn/'+outdir_name)
tol = 0.02
err_count = 0
err_idx = []
for idx in range(-1000,0):
  result_dir = os.path.join(exp_dir,"idx"+str(idx))
  with open(os.path.join(result_dir,'statistics.txt')) as json_file:
      [_,mse_x,mse_ynorm2] = json.load(json_file)
  if mse_ynorm2 > tol:
    print("error idx: ",idx,", ||y-A(x)||^2=",mse_ynorm2)
    err_count += 1
    err_idx.append(idx)
    ml_dir = os.path.join(result_dir,"mlRecon")
    if not os.path.exists(ml_dir):
      os.makedirs(ml_dir)
    z = x_train[idx]
    y = np.reshape(mnist_model.predict(np.reshape(x_train[idx],(1,28,28))), (10,))
    x = x_train[idx]
    for itr in range(50):
      Ax = np.reshape(mnist_model.predict(x.reshape((1,28,28))), (10,))
      H = pseudo_prox_map_nonlinear(np.subtract(y,Ax),x,dpm_model)
      x = np.clip(np.add(x, H),0,1)
      
      fig, ax = plt.subplots()
      cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
      im = ax.imshow(x, cmap='coolwarm',vmin=0,vmax=1)
      fig.colorbar(im, cax=cax, orientation='horizontal')
      plt.savefig(os.path.join(ml_dir,'x_itr'+str(itr)+'.png'))
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05])
    im = ax.imshow(y.reshape((1,10)), cmap='coolwarm',vmin=-1,vmax=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(ml_dir,'y.png'))
print("total error cases: ",err_count)
print(err_idx)
