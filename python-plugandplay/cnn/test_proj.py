import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

model_name = "mnist_projection"
json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
prj_model = model_from_json(loaded_model_json)
# load weights into new model
prj_model.load_weights(model_name+'.h5')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float64')
x_train = x_train.astype('float64')
x_test /= 255.
x_train /= 255.
x_gd = x_train[-20]
pj_mtx = [x_gd.sum(axis=0),x_gd.sum(axis=1)]

x_hat = prj_model.predict(np.reshape(pj_mtx,(1,56)))[0]
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(x_gd, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig("x_gd.png")
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(x_hat, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig("x_hat.png")

