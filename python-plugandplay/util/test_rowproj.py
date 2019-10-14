from proj import rowproj, colproj
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
from math import sqrt
import os

siglambd = 0.1
sigy = 0.1
output_dir = "./"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
np.random.seed(2019)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float64')
x_train = x_train.astype('float64')
x_test /= 255.
x_train /= 255.
x_gd = x_train[-20]
yr = x_gd.sum(axis=1)
#yr += np.random.normal(0,sigy,yr.shape)
yc = x_gd.sum(axis=0)
#yc += np.random.normal(0,sigy,yc.shape)
#v = x_gd+np.random.normal(0,siglambd,x_gd.shape)
# plot yr and yc
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(yr.reshape((1,28)), cmap='coolwarm',vmin=0,vmax=28.)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join(output_dir,"yr.png"))
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(yc.reshape((1,28)), cmap='coolwarm',vmin=0,vmax=28.)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join(output_dir,"yc.png"))
# plot v
v = np.zeros((28,28))
#v = copy.deepcopy(x_gd)
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(v, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join(output_dir,"v.png"))
  
xr = rowproj(v,yr,siglambd,sigy)
xc = colproj(v,yc,siglambd,sigy)
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(xc, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join(output_dir,"Fv_c.png"))
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(xr, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join(output_dir,"Fv_r.png"))

