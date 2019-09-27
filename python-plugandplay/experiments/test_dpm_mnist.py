import numpy as np
import sys,os
from math import sqrt
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imsave
import matplotlib.pyplot as plt
from feed_model import pseudo_prox_map_nonlinear
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.models import model_from_json
import copy

sigw = 0.
sigTrain = "var"
_log_data = False
sigTest = 0.
datagen_method = "mnist_mixed"
rows = 28
cols = 28
perterb = 1e-30

# read DPM model for mnist classifier
dpm_model_dir=os.path.join(os.getcwd(),'../cnn/dpm_model_mnist/'+datagen_method)
if _log_data:
  dpm_model_name = "model_mixed3_log_mnist_sig_"+str(sigTrain)+"_sigw"+str(sigw)
else:
  dpm_model_name = "model_mixed3_mnist_sig_"+str(sigTrain)+"_sigw"+str(sigw)
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

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x = x_train[-20]/255.
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(x, cmap='coolwarm',vmin=x.min(),vmax=x.max())
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig('x_gd.png')

np.random.seed(2019)
epsil = np.random.normal(0,sigTest,(rows,cols))
v = x + epsil
y = np.reshape(mnist_model.predict(v.reshape((1,rows,cols))), (10,))

Av = np.reshape(mnist_model.predict(v.reshape((1,rows,cols))), (10,))
Ax = np.reshape(mnist_model.predict(x.reshape((1,rows,cols))), (10,))
y_Av = np.reshape(y-Av, (1,10))
y_Ax = np.reshape(y-Ax, (1,10))
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(y_Ax, cmap='coolwarm',vmin=-1,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig('y-Ax.png')

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(np.reshape(y,(1,10)), cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig('y.png')

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(v, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.title("v")
plt.savefig('v.png')
plt.close()
if _log_data:
  H = pseudo_prox_map_nonlinear(np.subtract(np.log10(y+perterb),np.log10(Av+perterb)),v,dpm_model)
else:
  H = pseudo_prox_map_nonlinear(y-Ax,x,dpm_model)
Fy_v = H + x
target = epsil
err = target - H

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(H, cmap='coolwarm',vmin=-target.max(),vmax=target.max())
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.title("$H(y-A(v);v)$")
plt.savefig('H.png')

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(err, cmap='coolwarm',vmin=-err.max(),vmax=err.max())
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.title("$-\epsilon-H(y-A(v);v)$")
plt.savefig('err.png')

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(Fy_v, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.title("$F_y(v)$")
plt.savefig('Fy_v.png')

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(target, cmap='coolwarm',vmin=-target.max(),vmax=target.max())
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.title("target")
plt.savefig('target.png')
plt.close()

