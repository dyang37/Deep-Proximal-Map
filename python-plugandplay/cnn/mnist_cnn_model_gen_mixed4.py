import tensorflow as tf
from keras.models import Model,Sequential, model_from_json
from keras.layers import Input,Reshape, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import pickle
import copy

model_name = "mnist_forward_cnn"
_train = False
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train /= 255.
x_test /= 255.
model = Sequential()
model.add(Reshape((28,28,1),input_shape=(28,28)))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
if _train:
  model.fit(x=x_test,y=y_test, epochs=100)
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
test_loss = loaded_model.evaluate(x_train, y_train)
print("test loss: ",test_loss)

print("Start generating training triplets")
sigw = 0.
yAv = []
v = []
epsil = []
n_samples = x_train.shape[0]
datagen_method = "mnist_mixed"
np.random.seed(2019)
for v_img in x_train[0:n_samples//4]:
  #sig = np.random.uniform(0,0.2)
  sig = min(abs(np.random.laplace(scale=0.1)), 0.5)
  epsil_k = np.random.normal(0,sig,v_img.shape)
  x_img = v_img+epsil_k
  y = loaded_model.predict(x_img.reshape((1,)+x_img.shape))
  y += np.random.normal(0,sigw,y.shape)
  Av = loaded_model.predict(v_img.reshape((1,)+v_img.shape))
  y_Av_k = y-Av
  yAv.append(y_Av_k)
  v.append(v_img)
  epsil.append(epsil_k)
for x_img in x_train[n_samples//4:2*n_samples//4]:
  #sig = np.random.uniform(0,0.2)
  sig = min(abs(np.random.laplace(scale=0.1)), 0.5)
  epsil_k = np.random.normal(0,sig,x_img.shape)
  v_img = x_img - epsil_k
  y = loaded_model.predict(x_img.reshape((1,)+x_img.shape))
  y += np.random.normal(0,sigw,y.shape)
  Av = loaded_model.predict(v_img.reshape((1,)+v_img.shape))
  y_Av_k = y-Av
  yAv.append(y_Av_k)
  v.append(v_img)
  epsil.append(epsil_k)
x_idx_list = np.random.choice(range(n_samples*2//4,n_samples*3//4), size=n_samples//4, replace=False)
v_idx_list = np.random.choice(range(n_samples*2//4,n_samples*3//4), size=n_samples//4, replace=False)
for (x_idx,v_idx) in zip(x_idx_list,v_idx_list):
  x_img = x_train[x_idx]
  v_img = x_train[v_idx]
  epsil_k = x_img - v_img
  y = loaded_model.predict(x_img.reshape((1,)+x_img.shape))
  y += np.random.normal(0,sigw,y.shape)
  Av = loaded_model.predict(v_img.reshape((1,)+v_img.shape))
  y_Av_k = y-Av
  yAv.append(y_Av_k)
  v.append(v_img)
  epsil.append(epsil_k)

for v_img in x_train[3*n_samples//4:n_samples-1000]:
  epsil.append(np.zeros((28,28)))
  yAv.append(np.zeros((1,10)))
  v.append(v_img)

yAv = np.reshape(yAv,(-1,10))
print("shape of v: ",np.shape(v))
print("shape of epsil: ",np.shape(epsil))
print("shape of y-Av: ",np.shape(yAv))
dict_name = '/root/datasets/'+datagen_method+'/mnist_cnn_mixed4_triplets_laplace.dat'
print('dict_name = ',dict_name)
dataset = {'epsil':epsil,'y_Av':yAv, 'v':v}
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
