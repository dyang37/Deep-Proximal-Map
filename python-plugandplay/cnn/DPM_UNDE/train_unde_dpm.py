import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
from keras import layers,models
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from keras.optimizers import Adam
#from unet import unet

_train = True
model_name = "UNDE_resnet_mixed2_8cat"
Nx = 400
Nz = 300
filepath = '/home/yang1467/datasets/UNDE/newDPMdataSpeed_8_dict'
y_Av_tensor = []
v_tensor = []
epsil_tensor = []
for ndict in range(0,3):
  filename = filepath+str(ndict)+".mat"
  data = h5py.File(filename, 'r')
  y_Av = np.transpose(data["yAv_dict"], (3,2,1,0))
  v = np.transpose(data["v_dict"], (3,2,1,0))
  epsil = np.transpose(data["eps_dict"], (3,2,1,0))
  y_Av_tensor.append(y_Av)
  v_tensor.append(v)
  epsil_tensor.append(epsil)
y_Av = np.vstack(y_Av_tensor)
v = np.vstack(v_tensor)
epsil = np.vstack(epsil_tensor)
[n_geo,n_sample,n_trans,nt] = np.shape(y_Av)
y_Av_train = y_Av.reshape(-1,n_trans,nt)[:,:,1:]
v_train = v.reshape(-1,Nx,Nz)
epsil_train = epsil.reshape(-1,Nx,Nz)
print(y_Av_train.shape)
print(v_train.shape)
print(epsil_train.shape)

def cnnStack(x,dim,ker_size=(3,3),depth=3,upsamp=True):
  for _ in range(depth):
    x = layers.Conv2D(dim,ker_size,padding='same',activation='relu',kernel_initializer='he_normal')(x)
    if upsamp:
      x = layers.UpSampling2D((2,2))(x)
  return x

def residual_stack1D(x,n_chann=32, downsize=2):
  def residual_unit(y,_strides=1):
    shortcut_unit=y
    # 1x1 conv linear
    y = layers.Conv1D(n_chann, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='relu',kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv1D(n_chann, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='relu',kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    # add batch normalization
    y = layers.add([shortcut_unit,y])
    return y
  x = layers.Conv1D(n_chann, data_format='channels_first',kernel_size=1, padding='same',activation='linear',kernel_initializer="he_normal")(x)
  x = layers.BatchNormalization()(x)
  x = residual_unit(x)
  x = residual_unit(x)
  # maxpool for down sampling
  x = layers.MaxPooling1D(data_format='channels_first',pool_size=downsize)(x)
  return x

def residual_stack2D(x,n_chann=32):
  def residual_unit(y,_strides=1):
    shortcut_unit=y
    # 1x1 conv linear
    y = layers.Conv2D(n_chann, kernel_size=(5,5),data_format='channels_first',strides=_strides,padding='same',activation='relu',kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(n_chann, kernel_size=(5,5),data_format='channels_first',strides=_strides,padding='same',activation='relu',kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    # add batch normalization
    y = layers.add([shortcut_unit,y])
    return y
  x = layers.Conv2D(n_chann, data_format='channels_first',kernel_size=(1,1), padding='same',activation='linear',kernel_initializer="he_normal")(x)
  x = layers.BatchNormalization()(x)
  x = residual_unit(x)
  x = residual_unit(x)
  return x

# Construct DPM neural network architecture
in_shp_yAv = np.shape(y_Av_train)[1:]
in_shp_v = np.shape(v_train)[1:]
input_yAv = layers.Input(shape=in_shp_yAv) # input shape 10x6000
input_v = layers.Input(shape=in_shp_v) 
yAv = residual_stack1D(input_yAv, n_chann=20, downsize=1) # output shape 20x6000
yAv = residual_stack1D(yAv, n_chann=40) # output shape 40x3000
yAv = residual_stack1D(yAv, n_chann=80) # output shape 80 x 1500
yAv = residual_stack1D(yAv, n_chann=400, downsize=5) # output shape 400 x 300
yAv2D = layers.Reshape((1,Nx,Nz))(yAv)
v = layers.Reshape((1,Nx,Nz))(input_v)
v = residual_stack2D(v)
v = residual_stack2D(v)
v = residual_stack2D(v)
v = residual_stack2D(v)
v = residual_stack2D(v,n_chann=1)
H = layers.Concatenate(axis=1)([v,yAv2D])
H = residual_stack2D(H)
H = residual_stack2D(H)
H = residual_stack2D(H)
H = residual_stack2D(H)
H = layers.Conv2D(1,(1,1),padding='same',data_format = 'channels_first',activation='tanh',kernel_initializer="he_normal")(H)
H_out = layers.Reshape((Nx,Nz))(H)
model = models.Model(inputs=[input_yAv,input_v],outputs=H_out)
model = multi_gpu_model(model, gpus=2)
model.summary()

# Start training
batch_size = 8
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0005))
if _train:
  model.fit([y_Av_train,v_train], epsil_train, epochs=50, batch_size=batch_size,shuffle=True)
  model_json = model.to_json()
  with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights(model_name+".h5")
  print("model saved to disk")
