import os,sys
sys.path.append(os.path.join(os.getcwd(), "../util"))
from construct_forward_model import construct_forward_model
from sr_util import windowed_sinc
from icdwrapper import Pyicd
import numpy as np
from keras import layers, models
from keras.utils import multi_gpu_model
import pickle
import matplotlib.pyplot as plt

sig = 10./255.
sigw = 10./255.
K = 4
h = windowed_sinc(K)

dataset = pickle.load(open("pseudo-proximal-map-dict-small.dat",'rb'))

cutoff_idx = 800
x_train = dataset["x"][:cutoff_idx]
v_train = dataset["v"][:cutoff_idx]
y_train = dataset["y"][:cutoff_idx]
fv_train = dataset["fv"][:cutoff_idx]
gd_train = np.subtract(x_train,v_train)

x_test = dataset["x"][cutoff_idx:]
v_test = dataset["v"][cutoff_idx:]
y_test = dataset["y"][cutoff_idx:]
fv_test = dataset["fv"][cutoff_idx:]
gd_test = np.subtract(x_test,v_test)

n_test_samples = np.shape(x_test)[0]
rows_hr = np.shape(x_test)[1]
cols_hr = np.shape(x_test)[2]

in_shp_fv = list(fv_train.shape[1:])
in_shp_y = list(y_train.shape[1:])
shp_v = list(v_train.shape[1:])
K0 = shp_v[0]/in_shp_fv[0]
K1 = shp_v[1]/in_shp_fv[1]

print('fv training data shape: ',fv_train.shape)
print('y training data shape: ',y_train.shape)
# construct neural network graph
input_fv = layers.Input(shape=in_shp_fv)
input_y = layers.Input(shape=in_shp_y)
H=layers.Subtract()([input_y,input_fv])
H=layers.Conv2D(1,(5,5),activation='relu',padding='same')(H)
while (K0 > 1) or (K1 > 1):
  H=layers.Conv2D(1,(3,3),activation='relu',padding='same')(H)
  H=layers.UpSampling2D((2,2))(H)
  if K0 > 1:
    K0=K0//2
  if K1 > 1:
    K1=K1//2
H_out=layers.Conv2D(1,(3,3),activation='relu',padding='same')(H)
model = models.Model(inputs=[input_fv,input_y],output=[H_out])
model = multi_gpu_model(model, gpus=3)
model.summary()


# Start training
batch_size = 256
model.compile(loss='mean_squared_error',optimizer='adam')
if _train:
  history = model.fit([fv_train,y_train], gd_train, epochs=40, batch_size=batch_size,shuffle=True)
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights("model.h5")
  print("model saved to disk")
  plt.plot(np.sqrt(history.history['loss']))
  plt.title('Loss')
  plt.show()

# load model 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate test data
test_loss = loaded_model.evaluate([fv_test,y_test], gd_test)
print('test loss:', test_loss)

H_bar = loaded_model.predict([fv_test,y_test],batch_size=batch_size)
assert(np.shape(H_bar)==np.shape(v_test))
x_hat_test = np.add(H_bar,v_test)

# compare cost of CNN to cost of ICD
print('... evaluating proximal map cost for cnn and icd ...')
icd_cost_avg = 0
cnn_cost_avg = 0
for x_cnn,v_sample,y_sample in zip(x_hat_test[:50],v_test[:50],y_test[:50]):
  # Perform ICD update for current v_sample
  x_cnn = np.reshape(x_cnn,(rows_hr,cols_hr))
  v_sample = np.reshape(v_sample,(rows_hr,cols_hr))
  y_sample = np.reshape(y_sample,(rows_hr//K, cols_hr//K))
  x_icd = np.random.rand(rows_hr,cols_hr)
  icd_cpp = Pyicd(y_sample,h,K,1/(sig*sig),sigw);
  for itr in range(10):
    x_icd = np.array(icd_cpp.update(x_icd,v_sample))
  Gx_icd = construct_forward_model(x_icd,K,h,0)
  Gx_cnn = construct_forward_model(x_cnn,K,h,0)
  icd_cost_avg += sum(sum((Gx_icd-y_sample)**2))/(2*sigw*sigw) + sum(sum((x_icd-v_sample)**2))/(2*sig*sig)
  cnn_cost_avg += sum(sum((Gx_cnn-y_sample)**2))/(2*sigw*sigw) + sum(sum((x_cnn-v_sample)**2))/(2*sig*sig)

icd_cost_avg /= n_test_samples
cnn_cost_avg /= n_test_samples

print('icd average cost = ',icd_cost_avg)
print('cnn average cost = ',cnn_cost_avg)
