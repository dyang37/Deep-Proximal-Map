import numpy as np
import sys,os
sys.path.append(os.path.join(os.getcwd(), "../util"))
from math import sqrt
from mace_mnist import mace
import matplotlib.pyplot as plt
import json
from keras.models import model_from_json
import tensorflow as tf
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from DnCNN import DnCNN
################### hyperparameters
sigw = 0. # noise level
sig = "var"
outdir_name = "mace4_lineSearch"
optim = True


output_dir = os.path.join(os.getcwd(),'../results/mace_mnist/'+outdir_name)
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
########### begin loading Deep Learning Models
# load pre-trained mnist model
mnist_model_dir = os.path.join(os.getcwd(),'../cnn')
mnist_model_name = "mnist_forward_autoencoder"
json_file = open(os.path.join(mnist_model_dir, mnist_model_name+'.json'), 'r')
mnist_model_json = json_file.read()
json_file.close()
mnist_model = model_from_json(mnist_model_json)
mnist_model.load_weights(os.path.join(mnist_model_dir, mnist_model_name+".h5"))
# load Deep proximal map model
dpm_model_dir=os.path.join(os.getcwd(),'../cnn/dpm_model_mnist/mnist_mixed/')
dpm_model_name = "model_dense_mixed4_flatten_mnist_sig_"+str(sig)+"_sigw"+str(sigw)
json_file = open(os.path.join(dpm_model_dir, dpm_model_name+'.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
dpm_model = model_from_json(loaded_model_json)
dpm_model.load_weights(os.path.join(dpm_model_dir,dpm_model_name+'.h5'))
# load pre-trained denoiser model
denoiser_name = "model_150"
denoiser_dir="/root/Deep-Proximal-Map/python-plugandplay/denoisers/DnCNN/DnCNN/TrainingCodes/dncnn_keras/models/DnCNN_sigma13"
denoiser_model = DnCNN(depth=17,filters=64,image_channels=1,use_bnorm=True)
denoiser_model.load_weights(os.path.join(denoiser_dir,denoiser_name+'.hdf5'))
'''
# load pre-trained projection model
proj_model_dir=os.path.join(os.getcwd(),'../cnn/')
proj_model_name = "mnist_projection"
json_file = open(os.path.join(proj_model_dir, proj_model_name+'.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
proj_model = model_from_json(loaded_model_json)
proj_model.load_weights(os.path.join(proj_model_dir,proj_model_name+'.h5'))
'''
################# End loading Deep Learning Models
################# Prepare output result dir
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float64')
x_train /= 255.
idx = -20
y = np.reshape(mnist_model.predict(x_train[idx].reshape((1,784))), (10,))
yr = x_train[idx].mean(axis=1)
yc = x_train[idx].mean(axis=0)
z = x_train[idx].reshape((28,28))
d = y_train[idx]
print("target digit: ",d)

if optim:
  tol = 0.00001
  gr = (sqrt(5.)+1.)/2.
  a = 0.
  b = 0.95/2.
  c = b - (b-a)/gr
  d = a + (b-a)/gr
  diff = c-d
  while abs(diff) > tol:
    print("current diff = ",diff)
    beta_c = [0.5,c,c,0.95-2*c]
    beta_d = [0.5,d,d,0.95-2*d]
    if mace(z,y,yr,yc,beta_c,mnist_model,dpm_model,denoiser_model,output_dir,rho=0.8) < mace(z,y,yr,yc,beta_d,mnist_model,dpm_model,denoiser_model,output_dir,rho=0.8):
      b = d
    else:
      a = c
    c = b - (b-a)/gr
    d = a + (b-a)/gr
    diff = c-d
  w_opt = (b+a)/2.
else:
  w_opt = 0.05
beta_opt = [0.05,w_opt,w_opt,0.95-2*w_opt]
mse_opt = mace(z,y,yr,yc,beta_opt,mnist_model,dpm_model,denoiser_model,output_dir,rho=0.8,savefig=True)
print("optimal sqrt of mse: ",mse_opt)
