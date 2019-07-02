import os,sys
sys.path.append(os.path.join(os.getcwd(), "../util"))
import numpy as np
from keras import layers, models
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
# allow GPU growth
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

_train = True
print('training switch: ',_train)
model_name = 'pmap_model_mnist'
dict_name = '/root/datasets/mnist_triplets.dat'
dataset = pickle.load(open(dict_name,"rb"))
epsil = np.array(dataset['epsil'])
y_fv=np.array(dataset['y_fv'])
v = np.array(dataset['v'])

n_samples = np.shape(epsil)[0]
print('total number of samples: ',n_samples)
# Random Shuffle and training/test set selection
np.random.seed(2019)
n_train = 50000
train_idx = np.random.choice(range(0,n_samples), size=n_train, replace=False)
test_idx = list(set(range(0,n_samples))-set(train_idx))
epsil_train = epsil[train_idx]
yfv_train = y_fv[train_idx]
v_train = v[train_idx]

[rows_v, cols_v] = np.shape(v_train)[1:]
in_shp_yfv = np.shape(yfv_train)[1:]

print('fv-y training data shape: ',np.shape(yfv_train))

### construct neural network graph
input_yfv = layers.Input(shape=in_shp_yfv)
yfv_flatten = layers.Flatten()(input_yfv)
A_yfv = layers.Dense(64, activation='relu')(yfv_flatten)
A_yfv = layers.Dense(rows_v*cols_v, activation='relu')(A_yfv)
yfv_in = layers.Reshape((rows_v,cols_v,1))(A_yfv)
print("yfv_in layer shape: ",yfv_in._keras_shape)

input_v = layers.Input(shape=(rows_v, cols_v))
v_in = layers.Reshape((rows_v,cols_v,1))(input_v)

input_stack = layers.concatenate([yfv_in,v_in],axis=-1)
print("stacked layer shape: ",input_stack._keras_shape)
n_channels = 8
H = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(input_stack)
for _ in range(2):
  H = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(H)
H = layers.Conv2D(1,(3,3),activation='tanh',padding='same',data_format='channels_last')(H)
H_out = layers.Reshape((rows_v,cols_v))(H)
model = models.Model(inputs=[input_yfv,input_v],output=H_out)
#model = multi_gpu_model(model, gpus=3)
model.summary()


# Start training
batch_size = 64
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001))
if _train:
  history = model.fit([yfv_train,v_train], epsil_train, epochs=100, batch_size=batch_size,shuffle=True)
  model_json = model.to_json()
  with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights(model_name+".h5")
  print("model saved to disk")
  plt.figure()
  plt.plot(np.sqrt(history.history['loss']))
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.title('training Loss')
  plt.savefig('loss.png')

# load model 
json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into model
loaded_model.load_weights(model_name+".h5")
print("Loaded model from disk")


# evaluate test data
yfv_test = y_fv[test_idx]
v_test = v[test_idx]
epsil_test = epsil[test_idx]

loaded_model.compile(loss='mean_squared_error',optimizer='adam')
test_loss = loaded_model.evaluate([yfv_test,v_test], epsil_test)
print('test loss:', test_loss)
