import os,sys,glob
from skimage.io import imread
sys.path.append(os.path.join(os.getcwd(), "../util"))
from construct_forward_model import construct_forward_model
from sr_util import windowed_sinc, gauss2D, avg_filt
from icdwrapper import Pyicd
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

def to_gray(x_in, K):
  [rows_in,cols_in] = np.shape(x_in)[0:2]
  rows_out = rows_in//K*K
  cols_out = cols_in//K*K
  x = np.zeros((rows_out, cols_out))
  for i in range(rows_out):
    for j in range(cols_out):
      r = x_in[i,j,0]
      g = x_in[i,j,1]
      b = x_in[i,j,2]
      x[i,j]=0.2989 * r + 0.5870 * g + 0.1140 * b
  return x

_train = True
print('training switch: ',_train)
sig = 0.2
sigw = 0.05
K = 4
h = windowed_sinc(K)
#h = gauss2D((33,33),1)
#h = avg_filt(9)
forward_name = 'sinc'
model_name = 'model_'+forward_name+'_noisy_simple'

epsil = []
y_fv=[]

n_samples = 0

for filename in glob.glob('/root/datasets/pmap/*/*.jpg'):
  #print(filename)
  n_samples += 1
  v_in = np.array(imread(filename), dtype=np.float32) / 255.0
  v_img = to_gray(v_in, K)
  x_img = np.random.normal(v_img,sig)
  epsil_k = np.subtract(x_img,v_img)
  y = construct_forward_model(x_img, K, h, 0)
  wk = np.random.normal(0,sigw,np.shape(y))
  fv = construct_forward_model(v_img, K, h, 0)
  y_fv_k = np.subtract(y,fv)
  assert (y_fv_k == (construct_forward_model(epsil_k, K, h, 0)+ wk)).all()
  epsil.append(epsil_k) 
  y_fv.append(y_fv_k)

epsil = np.array(epsil)
y_fv = np.array(fv_y)

# Random Shuffle and training/test set selection
np.random.seed(2019)
n_train = n_samples*8//10
train_idx = np.random.choice(range(0,n_samples), size=n_train, replace=False)
test_idx = list(set(range(0,n_samples))-set(train_idx))
epsil_train = epsil[train_idx]
yfv_train = y_fv[train_idx]

rows_hr = np.shape(epsil_train)[1]
cols_hr = np.shape(epsil_train)[2]
print('rows_hr=',rows_hr)
print('cols_hr=',cols_hr)
in_shp_yfv = np.shape(yfv_train)[1:]

print('fv-y training data shape: ',np.shape(yfv_train))

### construct neural network graph
input_yfv = layers.Input(shape=(rows_hr//K,cols_hr//K))
yfv_in = layers.Reshape(in_shp_yfv+(1,))(input_yfv)

n_channels = 16
y_fv_in = layers.Conv2D(n_channels,(3,3),activation='linear',padding='same')(yfv_in)
k = K
while (k > 1):
  y_fv_in = layers.Conv2D(n_channels,(3,3),activation='linear',padding='same')(y_fv_in)
  y_fv_in=layers.UpSampling2D((2,2))(fv_y_in)
  k /= 2

H = layers.Conv2D(n_channels, (3,3), padding='same',activation='linear')(H)
H = layers.Conv2D(1, (3,3), padding='same',activation='linear')(H)
H_out = layers.Reshape((rows_hr,cols_hr))(H)
model = models.Model(inputs=input_yfv,output=H_out)
model = multi_gpu_model(model, gpus=3)
model.summary()


# Start training
batch_size = 128
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001))
if _train:
  history = model.fit(yfv_train, epsil_train, epochs=100, batch_size=batch_size,shuffle=True)
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
print('shape of test data:', np.shape(yfv_test))
epsil_test = epsil[test_idx]

loaded_model.compile(loss='mean_squared_error',optimizer='adam')
test_loss = loaded_model.evaluate(yfv_test, epsil_test)
print('test loss:', test_loss)
