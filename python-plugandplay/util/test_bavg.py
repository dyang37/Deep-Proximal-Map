from proxMap import PM_blockavg
from forward_model import blockAvgModel
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
from math import sqrt
import os

L = 4
siglambd = 0.1
sigy = 0.1

np.random.seed(2019)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float64')
x_train /= 255.
x_gd = x_train[-20]
y = blockAvgModel(x_gd,L)
y += np.random.normal(0,sigy,y.shape)
v = x_gd + np.random.normal(0,siglambd,x_gd.shape)

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(y, cmap='coolwarm',vmin=0,vmax=1.)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig("y_bavg.png")

Fv = PM_blockavg(v,y,siglambd,sigy,L=L)

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(Fv, cmap='coolwarm',vmin=0,vmax=1.)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig("Fv_bavg.png")

