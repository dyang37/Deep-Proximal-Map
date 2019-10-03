from icdproj import icd_rowproj, icd_colproj
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
from math import sqrt
sigx = 0.1
sigy = 0.1

np.random.seed(2019)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float64')
x_train = x_train.astype('float64')
x_test /= 255.
x_train /= 255.
x_gd = x_train[-20]
yr = x_gd.sum(axis=1)
yr += np.random.normal(0,sigy,yr.shape)
yc = x_gd.sum(axis=0)
yc += np.random.normal(0,sigy,yc.shape)
#v = x_gd+np.random.normal(0,sigx,x_gd.shape)
v = np.zeros(x_gd.shape)
#v = copy.deepcopy(x_gd)
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(v, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig("v.png")
sig = sigx/sigy
x = np.random.rand(28,28)
#x = copy.deepcopy(x_gd)
map_cost = []
xcost = []
maxitr = 50
for itr in range(maxitr):
  print("itr ",itr)
  fig, ax = plt.subplots()
  cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
  im = ax.imshow(x, cmap='coolwarm',vmin=0,vmax=1)
  fig.colorbar(im, cax=cax, orientation='horizontal')
  plt.savefig("x_itr"+str(itr)+".png")
  map_cost.append(((yc-x.sum(axis=0))**2).sum(axis=None)/2. + ((x-v)**2).sum(axis=None)/(2*sig*sig) )
  xcost.append(sqrt(((x-x_gd)**2).mean(axis=None)))
  #x = np.clip(icd_rowproj(x,v,yr,sig=sig),0,1)
  x = np.clip(icd_colproj(x,v,yc,sig=sig),0,1)
  
plt.figure()
plt.semilogy(range(maxitr),map_cost)
plt.xlabel("iteration")
plt.ylabel("proximal map cost")
plt.savefig("MAPcost.png")
plt.figure()
plt.semilogy(range(maxitr),xcost)
plt.xlabel("iteration")
plt.ylabel("$\sqrt{\dfrac{1}{N}||x_{gd}-x||^2}$")
plt.savefig("xcost.png")
