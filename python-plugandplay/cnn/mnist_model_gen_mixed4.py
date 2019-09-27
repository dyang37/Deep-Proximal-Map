import tensorflow as tf
from keras.models import Model,Sequential, model_from_json
from keras.layers import Input,Reshape, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import pickle
import copy

model_name = "mnist_forward_autoencoder"
_train = True
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train /= 255.
x_test /= 255.
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))
print(x_train.shape)
print(x_test.shape)
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(10, activation='softmax')(encoded)
model = Model(input_img, decoded)
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
perterb = 1e-30
yAv = []
v = []
epsil = []
n_samples = x_train.shape[0]
datagen_method = "mnist_mixed"
np.random.seed(2019)
for v_img in x_train[0:n_samples//4]:
  sig = np.random.uniform(0,0.3)
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
  sig = np.random.uniform(0,0.3)
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

for v_img in x_train[3*n_samples//4:n_samples-20]:
  epsil.append(np.zeros((28*28,)))
  yAv.append(np.zeros((1,10)))
  v.append(v_img)

yAv = np.reshape(yAv,(-1,10))
print("shape of v: ",np.shape(v))
print("shape of y-Av: ",np.shape(yAv))
dict_name = '/root/datasets/'+datagen_method+'/mnist_mixed4_'
dict_name += 'flatten_'
dict_name += 'triplets_sigvar_sigw'+str(sigw)+'.dat'
print('dict_name = ',dict_name)
dataset = {'epsil':epsil,'y_Av':yAv, 'v':v}
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
