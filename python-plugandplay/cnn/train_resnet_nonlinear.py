import os,sys
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
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

# parameters
clip = True
sig = 0.05
sigw = 0.
sigma_g = 10
alpha = 0.5
gamma = 2.
forward_name = 'nonlinear'

_train = True
print('training switch: ',_train)
dict_name = '/root/datasets/raw_input.dat'
if clip:
  model_name = 'model_resnet_nonlinear_noiseless_clip'
else:
  model_name = 'model_resnet_nonlinear_noiseless_noclip'

if sigw == 0:
  dict_dir = '/root/datasets/noiseless_nonlinear_dict'
else:
  dict_dir = '/root/datasets/noisy_nonlinear_dict'


v = np.empty((0,512,512))
epsil = np.empty((0,512,512))
y_Av = np.empty((0,512,512))
for n_dict in range(0,3):
  dict_name = dict_dir+str(n_dict)+'.dat'
  dataset = pickle.load(open(dict_name,"rb"))
  v = np.append(v,dataset["v"],axis=0)  
  epsil = np.append(epsil,dataset["epsil"],axis=0)  
  y_Av = np.append(y_Av,dataset["y_Av"],axis=0)  

n_samples,rows,cols = v.shape

np.random.seed(2019)
n_train = n_samples//10*9
train_idx = np.random.choice(range(0,n_samples), size=n_train, replace=False)
test_idx = list(set(range(0,n_samples))-set(train_idx))
epsil_train = epsil[train_idx]
yAv_train = y_Av[train_idx]
v_train = v[train_idx]

in_shp_yAv = np.shape(yAv_train)[1:]
in_shp_v = np.shape(v_train)[1:]


def residual_stack(x, n_chann=8):
  def residual_unit(y,_strides=1):
    shortcut_unit=y
    # 1x1 conv linear
    y = layers.Conv2D(n_chann, (3,3),strides=_strides,padding='same',activation='linear')(y)
    y = layers.Conv2D(n_chann, (3,3),strides=_strides,padding='same',activation='linear')(y)
    y = layers.BatchNormalization()(y)
    # add batch normalization
    y = layers.add([shortcut_unit,y])
    return y

  x = layers.Conv2D(n_chann, (1,1), padding='same',activation='linear')(x)
  x = residual_unit(x)
  x = residual_unit(x)
  # maxpool for down sampling
  return x

### construct neural network graph
n_channels = 8
input_yAv = layers.Input(shape=(rows,cols))
input_v = layers.Input(shape=(rows,cols))
yAv_in = layers.Reshape(in_shp_yAv+(1,))(input_yAv)
v_in = layers.Reshape(in_shp_v+(1,))(input_v)
H_stack = layers.concatenate([yAv_in,v_in],axis=-1)
H = layers.Conv2D(n_channels,(3,3),activation='relu',padding='same')(H_stack)

for _ in range(3):
  H = residual_stack(H, n_chann=n_channels)
H_tanh = layers.Conv2D(1,(3,3),activation='tanh',padding='same')(H)
H_out = layers.Reshape((rows,cols))(H_tanh)
model = models.Model(inputs=[input_yAv,input_v],output=H_out)
model = multi_gpu_model(model, gpus=3)
model.summary()


# Start training
batch_size = 128
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001))
if _train:
  history = model.fit([yAv_train,v_train], epsil_train, epochs=100, batch_size=batch_size,shuffle=True)
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
