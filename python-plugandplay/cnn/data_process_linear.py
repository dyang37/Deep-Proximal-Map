import os,sys,glob
from skimage.io import imread
sys.path.append(os.path.join(os.getcwd(), "../util"))
from construct_forward_model import construct_forward_model
from sr_util import windowed_sinc, gauss2D, avg_filt
from icdwrapper import Pyicd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# allow GPU growth

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

sig = 0.2
sigw = 0.05
K = 4
#h = windowed_sinc(K)
h = gauss2D((33,33),1)
#h = avg_filt(9)
forward_name = 'gauss'
model_name = 'model_'+forward_name+'_noisy_simple'

epsil = []
y_fv=[]

n_samples = 0

for filename in glob.glob('/root/datasets/DIV2K_*_HR/*.png'):
  #print(filename)
  n_samples += 1
  v_in = np.array(imread(filename), dtype=np.float32) / 255.0
  v_img = to_gray(v_in, K)
  [rows_hr,cols_hr] = v_img.shape
  rows_ctr = rows_hr//2
  cols_ctr = cols_hr//2
  v = v_img[rows_ctr-256:rows_ctr+256,cols_ctr-256:cols_ctr+256]
  epsil_k = np.random.normal(0,sig,v.shape)
  x_img = np.add(v, epsil_k)
  y = construct_forward_model(x_img, K, h, sigw)
  fv = construct_forward_model(v, K, h, 0)
  y_fv_k = np.subtract(y,fv)
  epsil.append(epsil_k) 
  y_fv.append(y_fv_k)

epsil = np.array(epsil)
y_fv = np.array(y_fv)
dataset = {'epsil':epsil,'y_fv':y_fv}
dict_name = '/root/datasets/pmap_exp_'+forward_name+'_hr.dat'
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
