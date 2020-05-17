import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
from keras import layers,models
from keras.utils import multi_gpu_model
from keras.models import model_from_json
from keras.optimizers import Adam
from scipy.io import loadmat

_train = True
model_name = "UNDE_resnet_lossless_1.11"
Nx = 360
Nz = 260
'''
filepath = '/home/yang1467/datasets/UNDE/lossless/LosslessDiskdata-sigLaplace-2chann.mat'
data = h5py.File(filepath, 'r')
y_Av_train = np.transpose(data["yAv_dict"], (2,1,0))
y_Av_train = y_Av_train[:,:,:5200]
v_train = np.transpose(data["v_dict"], (3,0,2,1))
epsil_train = np.transpose(data["eps_dict"], (3,0,2,1))
filepath = '/home/yang1467/datasets/UNDE/lossless/LosslessDiskdata-sig0.1-2chann.mat'
data = h5py.File(filepath, 'r')
y_Av_train = np.transpose(data["yAv_dict"], (2,1,0))
v_train = np.transpose(data["v_dict"], (3,0,2,1))
epsil_train = np.transpose(data["eps_dict"], (3,0,2,1))
'''
filepath = '/home/yang1467/datasets/UNDE/lossless/LosslessDiskdata-zerodiff-2chann.mat'
data = h5py.File(filepath, 'r')
y_Av_train0 = np.transpose(data["yAv_dict"], (2,1,0))
v_train0 = np.transpose(data["v_dict"], (3,0,2,1))
epsil_train0 = np.transpose(data["eps_dict"], (3,0,2,1))
filepath = '/home/yang1467/datasets/UNDE/lossless/LosslessDiskdata-sig0.1-2chann.mat'
data = h5py.File(filepath, 'r')
y_Av_train1 = np.transpose(data["yAv_dict"], (2,1,0))
v_train1 = np.transpose(data["v_dict"], (3,0,2,1))
epsil_train1 = np.transpose(data["eps_dict"], (3,0,2,1))
filepath = '/home/yang1467/datasets/UNDE/lossless/LosslessDiskdata-Loopdiff-0.1var-zero-disk-2chann.mat'
data = h5py.File(filepath, 'r')
y_Av_train2 = np.transpose(data["yAv_dict"], (2,1,0))
y_Av_train2 = y_Av_train2[:3000,:,:]
#y_train = np.transpose(data["Ax_dict"], (2,1,0))
v_train2 = np.transpose(data["v_dict"], (3,0,2,1))
v_train2 = v_train2[:3000,:,:,:]
epsil_train2 = np.transpose(data["eps_dict"], (3,0,2,1))
epsil_train2 = epsil_train2[:3000,:,:,:]

y_Av_train = np.vstack((y_Av_train0,y_Av_train1,y_Av_train2))
v_train = np.vstack((v_train0,v_train1,v_train2))
epsil_train = np.vstack((epsil_train0,epsil_train1,epsil_train2))
print("Before index filtering")
print(y_Av_train.shape)
print(v_train.shape)
print(epsil_train.shape)
n_samples = y_Av_train.shape[0]
# rule out any invalid data
invalid_idx = [i for i in range(n_samples) if np.any(y_Av_train[i,:,:] != y_Av_train[i,:,:]) or np.any(y_Av_train[i,:,:]>100)]
valid_idx = [i for i in range(n_samples) if i not in invalid_idx]
v_train = v_train[valid_idx,:,:]
epsil_train = epsil_train[valid_idx,:,:,:]
#F_train = F_train[valid_idx,:,:,:]
y_Av_train = y_Av_train[valid_idx,:,:]
#y_train = y_train[valid_idx,:,:]
print("After index filtering")
print("after combination")
print(y_Av_train.shape)
print(v_train.shape)
print(epsil_train.shape)

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
yAv = residual_stack1D(input_yAv, n_chann=20, downsize=1) # output shape 20x5200
yAv = residual_stack1D(yAv, n_chann=40) # output shape 40x2600
yAv = residual_stack1D(yAv, n_chann=80) # output shape 80 x 1300
yAv = residual_stack1D(yAv, n_chann=360, downsize=5) # output shape 360 x 260
yAv2D = layers.Reshape((1,Nx,Nz))(yAv)
yAv2D = residual_stack2D(yAv2D)
yAv2D = residual_stack2D(yAv2D)
yAv2D = residual_stack2D(yAv2D)
yAv2D = residual_stack2D(yAv2D)
v = residual_stack2D(input_v)
v = residual_stack2D(v)
v = residual_stack2D(v)
v = residual_stack2D(v)
v = residual_stack2D(v)
H = layers.Concatenate(axis=1)([v,yAv2D])
H = residual_stack2D(H,n_chann=64)
H = residual_stack2D(H,n_chann=32)
H = residual_stack2D(H,n_chann=16)
H = residual_stack2D(H,n_chann=8)
H = residual_stack2D(H,n_chann=4)
H_out = layers.Conv2D(2,(1,1),padding='same',data_format = 'channels_first',activation='tanh',kernel_initializer="he_normal")(H)
model = models.Model(inputs=[input_yAv,input_v],outputs=H_out)
model = multi_gpu_model(model, gpus=3)
model.summary()

# Start training
batch_size = 12
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0005))
if _train:
  model.fit([y_Av_train,v_train], epsil_train, epochs=100, batch_size=batch_size,shuffle=True)
  model_json = model.to_json()
  with open(model_name+".json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights(model_name+".h5")
  print("model saved to disk")
