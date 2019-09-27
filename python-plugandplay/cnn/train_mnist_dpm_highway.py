import os,sys
sys.path.append(os.path.join(os.getcwd(), "../util"))
import numpy as np
from keras import layers, models
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from keras.optimizers import Adam
import pickle
from skimage.io import imread
import matplotlib.pyplot as plt
# allow GPU growth
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import keras.backend as K

_train = True
_log_data = False
print('training switch: ',_train)

sig = "var"
sigw = 0.
datagen_method = "mnist_mixed"


if _log_data:
  model_name = "dpm_model_mnist/"+datagen_method+"/model_highway_mixed4_flatten_log_mnist_sig_"+str(sig)+"_sigw"+str(sigw)
  dict_name = "/root/datasets/"+datagen_method+"/mnist_mixed4_flatten_log_triplets_sig"+str(sig)+"_sigw"+str(sigw)+".dat"
  dataset = pickle.load(open(dict_name,"rb"))
  y_Av = np.array(dataset['log_y_Av'])
else:
  model_name = "dpm_model_mnist/"+datagen_method+"/model_highway_mixed4_flatten_mnist_sig_"+str(sig)+"_sigw"+str(sigw)
  dict_name = "/root/datasets/"+datagen_method+"/mnist_mixed4_flatten_triplets_sig"+str(sig)+"_sigw"+str(sigw)+".dat"
  dataset = pickle.load(open(dict_name,"rb"))
  y_Av = np.array(dataset['y_Av'])
v = np.array(dataset['v'])
epsil = np.array(dataset['epsil'])
[n_samples,n_pixels] = np.shape(v)
print('total number of samples: ',n_samples)

# Random Shuffle and training/test set selection
np.random.seed(2019)
n_train = n_samples
train_idx = np.random.choice(range(0,n_samples), size=n_train, replace=False)
print(train_idx)
test_idx = list(set(range(0,n_samples))-set(train_idx))
epsil_train = epsil[train_idx]
yAv_train = y_Av[train_idx]
v_train = v[train_idx]


############## DMP Model Design
in_shp_y = np.shape(yAv_train)[1:]
in_shp_v = np.shape(v_train)[1:]
### construct neural network graph
input_yAv = layers.Input(shape=(10,))
input_v = layers.Input(shape=(n_pixels,))
yAv_decode = layers.Dense(16,activation='relu')(input_yAv)
yAv_decode = layers.Dense(32,activation='relu')(yAv_decode)
yAv_decode = layers.Dense(64,activation='relu')(yAv_decode)
yAv_decode = layers.Dense(128,activation='relu')(yAv_decode)
yAv_decode = layers.Dense(784,activation='relu')(yAv_decode)
R = layers.Reshape((28,28,1))(yAv_decode)
reshape_v = layers.Reshape((28,28,1))(input_v)
# define w(r) with tensorflow
def residual(r):
  w = K.mean(r*r)
  return w*r

def directpass(tensor_in):
  r = tensor_in[0]
  v = tensor_in[1]
  w = 1.-K.mean(r*r)
  return w*v

H1 = layers.Lambda(residual)(R)
#H2 = layers.Lambda(directpass)([R,reshape_v])
H = layers.Concatenate(axis=-1)([H1,reshape_v])
H = layers.Conv2D(32,(3,3),padding='same',activation='relu')(H)
H = layers.Conv2D(16,(3,3),padding='same',activation='relu')(H)
H = layers.Conv2D(8,(3,3),padding='same',activation='relu')(H)
H = layers.Conv2D(4,(3,3),padding='same',activation='relu')(H)
H = layers.Conv2D(2,(3,3),padding='same',activation='relu')(H)
H = layers.Conv2D(1,(3,3),padding='same',activation='tanh')(H)
H_out = layers.Flatten()(H)
model = models.Model(inputs=[input_yAv,input_v],output=H_out)
model = multi_gpu_model(model, gpus=3)
model.summary()
###############


# Start training
batch_size = 256
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0002))
if _train:
  history = model.fit([yAv_train,v_train], epsil_train, epochs=150, batch_size=batch_size,shuffle=True)
  model_json = model.to_json()
  with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights(model_name+".h5")
  print("model saved to disk")
  plt.figure()
  plt.semilogy(np.sqrt(history.history['loss']))
  plt.xlabel('epoch')
  plt.ylabel('$log\{mse\}$')
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
yAv_test = y_Av[test_idx]
v_test = v[test_idx]
epsil_test = epsil[test_idx]

loaded_model.compile(loss='mean_squared_error',optimizer='adam')
test_loss = loaded_model.evaluate([yAv_test,v_test], epsil_test)
print('test loss:', test_loss)
print("model name: ",model_name)
