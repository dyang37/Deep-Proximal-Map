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

_train = True
print('training switch: ',_train)

sig = 0.05
sigw = 0.05

dict_name = '/home/yang1467/deepProxMap/datasets/mnist_triplets_sigw'+str(sigw)+'.dat'
dataset = pickle.load(open(dict_name,"rb"))
v = np.array(dataset['v'])
y_Av = np.array(dataset['y_Av'])
epsil = np.array(dataset['epsil'])
[n_samples,rows,cols] = np.shape(v)
print('total number of samples: ',n_samples)

# parameters
forward_name = 'mnist'
model_name = "dpm_model_mnist/model_mnist_sigw_"
model_name += str(sigw)
print("model name: ",model_name)

# Random Shuffle and training/test set selection
np.random.seed(2019)
n_train = n_samples//10*9
train_idx = np.random.choice(range(0,n_samples), size=n_train, replace=False)
print(train_idx)
test_idx = list(set(range(0,n_samples))-set(train_idx))
epsil_train = epsil[train_idx]
yAv_train = y_Av[train_idx]
v_train = v[train_idx]

############## DMP Model Design
in_shp_yAv = np.shape(yAv_train)[1:]
in_shp_v = np.shape(v_train)[1:]
### construct neural network graph
n_channels = 32
input_yAv = layers.Input(shape=(10,))
input_v = layers.Input(shape=(rows,cols))
yAv = layers.Reshape(in_shp_yAv+(1,))(input_yAv)
v_in = layers.Reshape(in_shp_v+(1,))(input_v)
for _ in range(3):
  yAv = layers.Conv1D(rows,3,data_format="channels_first",activation='relu',padding='same')(yAv)
for _ in range(3):
  yAv = layers.Conv1D(cols,3,data_format="channels_last",activation='relu',padding='same')(yAv)
yAv2D = layers.Reshape((rows,cols,1))(yAv)
H_stack = layers.concatenate([yAv2D,v_in],axis=-1)
H = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(H_stack)

for _ in range(5):
  H = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(H)
H_tanh = layers.Conv2D(1,(3,3),activation='tanh',padding='same')(H)
H_out = layers.Reshape((rows,cols))(H_tanh)
model = models.Model(inputs=[input_yAv,input_v],output=H_out)
model = multi_gpu_model(model, gpus=3)
model.summary()
###############

# Start training
batch_size = 256
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0002))
if _train:
  history = model.fit([yAv_train,v_train], epsil_train, epochs=300, batch_size=batch_size,shuffle=True)
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
