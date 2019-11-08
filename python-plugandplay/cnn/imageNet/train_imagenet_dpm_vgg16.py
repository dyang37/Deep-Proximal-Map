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

sig = "var"
sigw = 0.
model_name = "dpm_model_imageNet/dpm_mixed4_imagenet_vgg16"
dict_name = "/root/datasets/DPM_VGG16_60000.dat"
dataset = pickle.load(open(dict_name,"rb"))

y_Av = np.array(dataset['y_Av'])
v = np.array(dataset['v'])
epsil = np.array(dataset['epsil'])
[n_samples,rows,cols,channs] = np.shape(v)
print('total number of samples: ',n_samples)
print("model name: ",model_name)

# Random Shuffle and training/test set selection
np.random.seed(2019)
n_train = n_samples
train_idx = np.random.choice(range(0,n_samples), size=n_train*9//10, replace=False)
print(train_idx)
test_idx = list(set(range(0,n_samples))-set(train_idx))
epsil_train = epsil[train_idx]
yAv_train = y_Av[train_idx]
v_train = v[train_idx]

def cnnStack(x,dim=32,ker_size=(3,3),depth=3,upsamp=True):
  for _ in range(depth):
    x = layers.Conv2D(dim,ker_size,padding='same',activation='relu')(x)
  if upsamp:
    x = layers.UpSampling2D((2,2))(x)
  return x

############## DPM Model Design
in_shp_y = np.shape(yAv_train)[1:]
in_shp_v = np.shape(v_train)[1:]
############## DPM neural network graph
input_yAv = layers.Input(shape=(1000,))
input_v = layers.Input(shape=(rows,cols,channs))
v_2D = layers.Reshape((224,224,1))(input_v)
yAv_dense = layers.Dense(1568,activation='relu')(input_yAv)
yAv_2D = layers.Reshape((7,7,32))(yAv_dense)
dim = 64
for _ in range(5):
  yAv_2D = cnnStack(yAv_2D,dim)
  dim //= 2
yAv_cnn = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu')(yAv_2D)
v_cnn = cnnStack(v_2D,64,depth=2,upsamp=False)
v_cnn = cnnStack(v_cnn,32,depth=2,upsamp=False)
v_cnn = cnnStack(v_2D,16,depth=2,upsamp=False)
v_cnn = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu')(v_cnn)
H = layers.concatenate([v_cnn,yAv_cnn])
H = layers.Conv2D(64,(5,5),padding='same',activation='relu')(H)
H = layers.Conv2D(32,(5,5),padding='same',activation='relu')(H)
H = layers.Conv2D(16,(5,5),padding='same',activation='relu')(H)
H = layers.Conv2D(8,(5,5),padding='same',activation='relu')(H)
H = layers.Conv2D(4,(5,5),padding='same',activation='relu')(H)
H = layers.Conv2D(2,(5,5),padding='same',activation='relu')(H)
H = layers.Conv2D(1,(5,5),padding='same',activation='tanh')(H)
H_out = layers.Reshape((224,224))(H)

############## END DPM neural network graph

model = models.Model(inputs=[input_yAv,input_v],output=H_out)
model = multi_gpu_model(model, gpus=3)
model.summary()
###############


# Start training
batch_size = 256
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0005))
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
