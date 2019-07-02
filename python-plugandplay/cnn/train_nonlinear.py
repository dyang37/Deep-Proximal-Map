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
from construct_forward_model import construct_nonlinear_model

clip = False

_train = True
print('training switch: ',_train)
dict_name = '/root/datasets/raw_input.dat'
if clip:
  model_name = 'model_nonlinear_noisy_clip'
else:
  model_name = 'model_nonlinear_noisy_noclip'
dataset = pickle.load(open(dict_name,"rb"))
v = np.array(dataset['v'])
print(np.shape(v))
[n_samples,rows,cols] = np.shape(v)
print('total number of samples: ',n_samples)

# parameters
clip = False
sig = 0.05
sigw = 0.05
sigma_g = 10
alpha = 0.5
gamma = 2.
forward_name = 'nonlinear'

epsil = []
y_Av = []
for v_img in v:
  epsil_k = np.random.normal(0,sig,v_img.shape)
  x_img = np.add(v_img, epsil_k)
  y = construct_nonlinear_model(x_img, sigma_g, alpha, sigw, gamma=gamma,clip=clip)
  Av = construct_nonlinear_model(v_img, sigma_g, alpha, 0, gamma=gamma,clip=clip)
  y_Av.append(y-Av)
  epsil.append(epsil_k) 
# Random Shuffle and training/test set selection
epsil  = np.array(epsil)
y_Av = np.array(y_Av)
np.random.seed(2019)
n_train = n_samples//10*9
train_idx = np.random.choice(range(0,n_samples), size=n_train, replace=False)
print(train_idx)
test_idx = list(set(range(0,n_samples))-set(train_idx))
epsil_train = epsil[train_idx]
yAv_train = y_Av[train_idx]
v_train = v[train_idx]

in_shp_yAv = np.shape(yAv_train)[1:]
in_shp_v = np.shape(v_train)[1:]

print('fv-y training data shape: ',np.shape(yAv_train))

### construct neural network graph
n_channels = 16
input_yAv = layers.Input(shape=(rows,cols))
input_v = layers.Input(shape=(rows,cols))
yAv_in = layers.Reshape(in_shp_yAv+(1,))(input_yAv)
v_in = layers.Reshape(in_shp_v+(1,))(input_v)
v_ = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(v_in)
for _ in range(5):
  v_ = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(v_)
v_ = layers.Conv2D(1,(3,3),activation='relu',padding='same')(v_)

H_stack = layers.concatenate([yAv_in,v_],axis=-1)
H = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(H_stack)

for _ in range(5):
  H = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(H)
H_tanh = layers.Conv2D(1,(3,3),activation='tanh',padding='same')(H)
H_out = layers.Reshape((rows,cols))(H_tanh)
model = models.Model(inputs=[input_yAv,input_v],output=H_out)
model = multi_gpu_model(model, gpus=3)
model.summary()


# Start training
batch_size = 128
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0001))
if _train:
  history = model.fit([yAv_train,v_train], epsil_train, epochs=200, batch_size=batch_size,shuffle=True)
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
yAv_test = y_Av[test_idx]
v_test = v[test_idx]
epsil_test = epsil[test_idx]

loaded_model.compile(loss='mean_squared_error',optimizer='adam')
test_loss = loaded_model.evaluate([yAv_test,v_test], epsil_test)
print('test loss:', test_loss)
