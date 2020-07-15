import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import pickle
import numpy as np
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from smallModel import cnnModel
from ptwspdModel import Model19
######## model name goes HERE
model_name = "UNDE_resnet_lossless_1.21_looped3000_gaussian3000-ptwsModel"
######## load pre-generated training data
'''
filepath = '/home/yang1467/datasets/UNDE/lossless/LosslessDiskdata-sig0.1-2chann.mat'
data = h5py.File(filepath, 'r')
y_Av_train0 = np.transpose(data["yAv_dict"], (2,1,0))
y_Av_train0 = y_Av_train0[:3000,:,:]
v_train0 = np.transpose(data["v_dict"], (3,0,2,1))
v_train0 = v_train0[:3000,:,:,:]
epsil_train0 = np.transpose(data["eps_dict"], (3,0,2,1))
epsil_train0 = epsil_train0[:3000,:,:,:]
print("gaussian data: ",np.shape(v_train0))
'''
filepath = '/home/yang1467/datasets/UNDE/lossless/artifact-data-#400.pkl'
file = open(filepath,"rb")
data = pickle.load(file)
y_Av_train0 = np.array(data["yAv"])
v_train0 = np.array(data["v"])
epsil_train0 = np.array(data["epsil"])
print("artifact diff data: ",np.shape(v_train0))
filepath = '/home/yang1467/datasets/UNDE/lossless/LosslessDiskdata-Loopdiff-0.1var-zero-disk-2chann.mat'
data = h5py.File(filepath, 'r')
y_Av_train1 = np.transpose(data["yAv_dict"], (2,1,0))
v_train1 = np.transpose(data["v_dict"], (3,0,2,1))
epsil_train1 = np.transpose(data["eps_dict"], (3,0,2,1))
print("looped diff data: ",np.shape(v_train1))
######## concatenate all data together
y_Av_train = np.vstack((y_Av_train0,y_Av_train1))
v_train = np.vstack((v_train0,v_train1))
epsil_train = np.vstack((epsil_train0,epsil_train1))
######### filter out invalid data
print("Before index filtering")
print(y_Av_train.shape)
print(v_train.shape)
print(epsil_train.shape)
n_samples = y_Av_train.shape[0]
invalid_idx = [i for i in range(n_samples) if np.any(y_Av_train[i,:,:] != y_Av_train[i,:,:]) or np.any(y_Av_train[i,:,:]>100)]
valid_idx = [i for i in range(n_samples) if i not in invalid_idx]
v_train = v_train[valid_idx,:,:]
epsil_train = epsil_train[valid_idx,:,:,:]
y_Av_train = y_Av_train[valid_idx,:,:]
print("After index filtering")
print("after combination")
print(y_Av_train.shape)
print(v_train.shape)
print(epsil_train.shape)

########## Training network goes HERE
model = cnnModel()
model = multi_gpu_model(model, gpus=3)
model.summary()

# Start training
batch_size = 60
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0005))
model.fit([y_Av_train,v_train], epsil_train, epochs=200, batch_size=batch_size,shuffle=True)
model_json = model.to_json()
with open(model_name+".json", "w") as json_file:
  json_file.write(model_json)
model.save_weights(model_name+".h5")
print("model saved to disk")
