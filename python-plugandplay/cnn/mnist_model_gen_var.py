import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Reshape, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import pickle
import copy
import random

model_name = "mnist_forward_model"
_train = False
_log_data = True
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
input_shape = (28, 28)
model = Sequential()
model.add(Reshape((28,28,1),input_shape=input_shape))
model.add(Conv2D(28, kernel_size=(8,4)))
model.add(MaxPooling2D(pool_size=(4, 6)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
if _train:
  model.fit(x=x_train,y=y_train, epochs=100)
  model_json = model.to_json()
  with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights(model_name+".h5")
    print("model saved to disk")

# load model
json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into model
loaded_model.load_weights(model_name+".h5")
print("Loaded model from disk")
loaded_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
test_loss = loaded_model.evaluate(x_test, y_test)
print("test loss: ",test_loss)

print("Start generating training triplets")
sigw = 0.05
perterb = 1e-30
yAv = []
v = []
epsil = []
n_samples = x_train.shape[0]
datagen_method = "mnist_mixed"
idx = 0
for v_img in x_train:
  sig = random.uniform(0,0.3)
  epsil_k = np.random.normal(0,sig,v_img.shape)
  if idx < n_samples//2:
    x_img = v_img+epsil_k
  else:
    x_img = copy.deepcopy(v_img)
    v_img = x_img - epsil_k
  y = loaded_model.predict(x_img.reshape((1,)+x_img.shape))
  y += np.random.normal(0,sigw,y.shape)
  Av = loaded_model.predict(v_img.reshape((1,)+v_img.shape))
  if _log_data:
    y_Av_k = np.log10(y+perterb) - np.log10(Av+perterb)
  else:
    y_Av_k = y-Av
  yAv.append(y_Av_k)
  v.append(v_img)
  epsil.append(epsil_k)
  idx += 1 

yAv = np.reshape(yAv,(n_samples,10))
print("shape of v: ",np.shape(v))
print("shape of y-Av: ",np.shape(yAv))
if _log_data:
  dataset = {'epsil':epsil,'log_y_Av':yAv, 'v':v}
  dict_name = '/root/datasets/'+datagen_method+'/mnist_log_triplets_sigvar_sigw'+str(sigw)+'.dat'
else:
  dataset = {'epsil':epsil,'y_Av':yAv, 'v':v}
  dict_name = '/root/datasets/'+datagen_method+'/mnist_triplets_sigvar_sigw'+str(sigw)+'.dat'
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
