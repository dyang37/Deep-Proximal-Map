import os,sys
import numpy as np
from forward_model import downsampling_model
from skimage.io import imread,imsave
from icdwrapper import Pyicd
from sr_util import windowed_sinc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import model_from_json
import tensorflow as tf
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from feed_model import pseudo_prox_map


K = 4
sigw = 0.05
sig = 0.2
lambd = 1./(sig*sig)
fig_in = 'test_gray'
z = np.array(imread('../'+fig_in+'.png'), dtype=np.float32) / 255.0
[rows_hr,cols_hr] = np.shape(z)
h = windowed_sinc(K)
y = downsampling_model(z, K, h, sigw)

# read pre-trained model for pseudo-proximal map
model_dir=os.path.join(os.getcwd(),'../cnn/linear_model')
model_name = "model_sinc_noisy_simple_hr"
json_file = open(os.path.join(model_dir, model_name+'.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join(model_dir,model_name+'.h5'))
#v = z + np.random.normal(0,sig,(rows_hr,cols_hr))
v = np.zeros((rows_hr,cols_hr))
Av = downsampling_model(v, K, h, 0)
# conventional PM output estimated by ICD
icd_cpp = Pyicd(y,h,K,lambd,sigw)
x_icd = np.random.rand(rows_hr,cols_hr)
for itr in range(10):
  x_icd = np.clip(icd_cpp.update(x_icd,v),0,None)
F_icd = x_icd
# deep PM output
H = pseudo_prox_map(y-Av, model)
F_cnn = H + v

imsave("y.png",np.clip(y,0,1))
imsave("v.png",np.clip(v,0,1))
imsave('F_dpm.png',np.clip(F_cnn,0,1))
imsave('F_cpm.png',np.clip(F_icd,0,1))

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(F_icd-F_cnn, cmap='coolwarm',vmin=-0.1,vmax=0.1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig('err_F.png')
