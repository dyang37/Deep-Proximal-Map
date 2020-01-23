import os,sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import h5py
import numpy as np
from keras import layers,models
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from keras.optimizers import Adam
#from unet import unet

_train = True
model_name = "UNDE_cnn_mixed2_8cat"
Nx = 400
Nz = 300
filepath = '/scratch/gilbreth/yang1467/DPMdataSpeed_8.mat'
data = h5py.File(filepath, 'r')
y_Av = np.transpose(data["yAv_dict"], (3,2,1,0))
v = np.transpose(data["v_dict"], (3,2,1,0))
epsil = np.transpose(data["eps_dict"], (3,2,1,0))
[n_geo,n_sample,n_trans,nt] = np.shape(y_Av)
y_Av = y_Av.reshape(-1,n_trans,nt)[:,:,:9000]
y_Av = np.transpose(y_Av,(0,2,1))
v = v.reshape(-1,Nx,Nz)
epsil = epsil.reshape(-1,Nx,Nz)
print(y_Av.shape)
print(v.shape)
print(epsil.shape)
# Random Shuffle and training/test set selection
np.random.seed(2020)
n_train = n_sample*n_geo
train_idx = np.random.choice(range(0,n_train), size=n_train, replace=False)
epsil_train = epsil[train_idx]
yAv_train = y_Av[train_idx]
v_train = v[train_idx]
def cnnStack(x,dim,ker_size=(3,3),depth=3,upsamp=True):
  for _ in range(depth):
    x = layers.Conv2D(dim,ker_size,padding='same',activation='relu',kernel_initializer='he_normal')(x)
    if upsamp:
      x = layers.UpSampling2D((2,2))(x)
  return x
# Construct DPM neural network architecture
in_shp_yAv = np.shape(yAv_train)[1:]
in_shp_v = np.shape(v_train)[1:]
input_yAv = layers.Input(shape=in_shp_yAv)
input_v = layers.Input(shape=in_shp_v)
yAv_dense = layers.Dense(12000)(input_yAv)
yAv_dense = layers.Permute((2,1))(yAv_dense)
print(yAv_dense.shape)
yAv_dense = layers.Reshape((12000,9000,1))(yAv_dense)
yAv_cnn = layers.MaxPooling2D(pool_size=(5,5))(yAv_dense)
yAv_cnn = layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer = 'he_normal')(yAv_cnn)
yAv_cnn = layers.MaxPooling2D(pool_size=(3,3))(yAv_cnn)
yAv_cnn = layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer = 'he_normal')(yAv_cnn)
yAv_cnn = layers.MaxPooling2D(pool_size=(2,2))(yAv_cnn)
yAv_cnn = layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer = 'he_normal')(yAv_cnn)
#yAv_cnn = unet(yAv_cnn)
yAv_cnn = cnnStack(yAv_cnn,64,depth=3,upsamp=False)
yAv_cnn = cnnStack(yAv_cnn,64,depth=3,upsamp=False)
yAv_cnn = cnnStack(yAv_cnn,64,depth=3,upsamp=False)
v_cnn = layers.Reshape((Nx,Nz,1))(input_v)
#v_cnn = unet(v_cnn)
v_cnn = cnnStack(v_cnn,64,depth=3,upsamp=False)
v_cnn = cnnStack(v_cnn,64,depth=3,upsamp=False)
v_cnn = cnnStack(v_cnn,64,depth=3,upsamp=False)
H = layers.concatenate([v_cnn,yAv_cnn])
#H = unet(H)
H = layers.Conv2D(128,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(H)
H = layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(H)
H = layers.Conv2D(32,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(H)
H = layers.Conv2D(16,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(H)
H = layers.Conv2D(8,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(H)
H = layers.Conv2D(4,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(H)
H = layers.Conv2D(2,(3,3),padding='same',activation='relu',kernel_initializer='he_normal')(H)
H = layers.Conv2D(1,(3,3),padding='same',activation='tanh',kernel_initializer='he_normal')(H)
H_out = layers.Reshape((Nx,Nz))(H)
model = models.Model(inputs=[input_yAv,input_v],output=H_out)
model = multi_gpu_model(model, gpus=2)
model.summary()

# Start training
batch_size = 8
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0005))
if _train:
  model.fit([yAv_train,v_train], epsil_train, epochs=50, batch_size=batch_size,shuffle=True)
  model_json = model.to_json()
  with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights(model_name+".h5")
  print("model saved to disk")
