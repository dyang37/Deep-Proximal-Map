import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Reshape, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import pickle

model_name = "mnist_forward_model"
_train = False

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
  model.fit(x=x_train,y=y_train, epochs=20)
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
sigw = 0
sig = 0.2
yfv = []
v = []
epsil = []
n_sample = 0
for v_img in x_train:
  fv = loaded_model.predict(v_img.reshape((1,)+v_img.shape))
  epsil_k = np.random.normal(0,sig,v_img.shape)
  x_img = np.add(v_img, epsil_k)
  y = loaded_model.predict(x_img.reshape((1,)+x_img.shape))
  y_fv_k = np.subtract(y,fv)
  yfv.append(y_fv_k)
  v.append(v_img)
  epsil.append(epsil_k)

print("shape of v: ",np.shape(v))
print("shape of y-fv: ",np.shape(yfv))
dataset = {'epsil':epsil,'y_fv':yfv, 'v':v}
dict_name = '/root/datasets/mnist_triplets.dat'
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
