import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float64')
x_train = x_train.astype('float64')
x_test /= 255.
x_train /= 255.
x1 = x_gd = x_train[-20]

for i in range(100):
  if y_train[i] == 3:
    x2 = x_train[i]
    break

x_avg = np.clip((x1+x2)/2.,0,1)
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(x1, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig("x1.png")
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(x_avg, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig("x_avg.png")
