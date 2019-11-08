import numpy as np
import sys,os
sys.path.append(os.path.join(os.getcwd(), "../util"))
from math import sqrt
from mace_mnist import mace
import matplotlib.pyplot as plt
import json
from keras.models import model_from_json
from forward_model import blockAvgModel
import tensorflow as tf
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from DnCNN import DnCNN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
################### hyperparameters
L = 4
optim = False
########### begin loading Deep Learning Models
# load pre-trained mnist model
mnist_model_dir = os.path.join(os.getcwd(),'../cnn')
mnist_model_name = "mnist_forward_cnn"
json_file = open(os.path.join(mnist_model_dir, mnist_model_name+'.json'), 'r')
mnist_model_json = json_file.read()
json_file.close()
mnist_model = model_from_json(mnist_model_json)
mnist_model.load_weights(os.path.join(mnist_model_dir, mnist_model_name+".h5"))
# load Deep proximal map model
dpm_model_dir=os.path.join(os.getcwd(),'../cnn/dpm_model_mnist/mnist_mixed/')
dpm_model_name = "model_cnn_mixed4_mnist_cnn_laplace"
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
################# End loading Deep Learning Models
################# Prepare output result dir
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float64')
x_train /= 255.
rmse_x_arr= []
norm2_y_arr= []
n_samples = 1000
print("total number of samples: ",n_samples)
for idx in range(-20,-10):
  print("idx ",idx)
  outdir_name = "MACE5_linesearch"
  exp_dir = os.path.join(os.getcwd(),'../results/mace_mnist_cnn/'+outdir_name)
  output_dir = os.path.join(exp_dir,"idx"+str(idx))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  z = x_train[idx]
  y = np.reshape(mnist_model.predict(np.reshape(x_train[idx],(1,28,28))), (10,))
  #y = [0.]*10
  #y[3] = 1.
  #y = np.array(y)
  yr = x_train[idx].mean(axis=1)
  yc = x_train[idx].mean(axis=0)
  yb = blockAvgModel(x_train[idx],L=L)
  d = y_train[idx]
  print("target digit: ",d)
  if optim:
    tol = 0.00001
    gr = (sqrt(5.)+1.)/2.
    a = 0.
    b = 0.8
    c = b - (b-a)/gr
    d = a + (b-a)/gr
    diff = c-d
    while abs(diff) > tol:
      print("current diff = ",diff)
      beta_c = [0.1,0.05,0.05,0.8-c,c]
      beta_d = [0.1,0.05,0.05,0.8-d,d]
      if mace(z,y,yr,yc,yb,L,beta_c,mnist_model,dpm_model,denoiser_model,output_dir,rho=0.8) < mace(z,y,yr,yc,yb,L,beta_d,mnist_model,dpm_model,denoiser_model,output_dir,rho=0.8):
        b = d
      else:
        a = c
      c = b - (b-a)/gr
      d = a + (b-a)/gr
      diff = c-d
    w_opt = (b+a)/2.
    beta_opt = [0.1,0.05,0.05,0.8-w_opt,w_opt]
  else:
    beta_opt = [0.1,0.05,0.05,0.75,0.05]
  (rmse_x,norm2_y) = mace(z,y,yr,yc,yb,L,beta_opt,mnist_model,dpm_model,denoiser_model,output_dir,rho=0.8,savefig=True)
  rmse_x_arr.append(rmse_x)
  norm2_y_arr.append(norm2_y)
  print("optimal sqrt of mse: ",rmse_x)
  #print("optimal weights: ",beta_opt)
print("average mse of x-x_gd = ",np.mean(rmse_x_arr))
print("average norm2 of y-Ax = ",np.mean(norm2_y_arr))
#with open(os.path.join(exp_dir,'statistics_all.txt'), 'w') as filehandle:
#  json.dump([np.mean(rmse_x_arr),np.mean(norm2_y_arr).item()], filehandle)
