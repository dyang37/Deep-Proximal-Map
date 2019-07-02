import os,sys,glob
from skimage.io import imread
sys.path.append(os.path.join(os.getcwd(), "../util"))
from construct_forward_model import construct_nonlinear_model
from sr_util import windowed_sinc, gauss2D, avg_filt
from icdwrapper import Pyicd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# allow GPU growth

def to_gray(x_in):
  [rows,cols] = np.shape(x_in)[0:2]
  x = np.zeros((rows, cols))
  for i in range(rows):
    for j in range(cols):
      r = x_in[i,j,0]
      g = x_in[i,j,1]
      b = x_in[i,j,2]
      x[i,j]=0.2989 * r + 0.5870 * g + 0.1140 * b
  return x

clip = False
sig = 0.2
sigw = 0.1
sigma = 10
alpha = 0.5
gamma = 2.

forward_name = 'nonlinear'

epsil = []
y_fv=[]
v_list = []
n_samples = 0

for filename in glob.glob('/root/datasets/DIV2K_*_HR/*.png'):
  #print(filename)
  n_samples += 1
  v_in = np.array(imread(filename), dtype=np.float32) / 255.0
  v_img = to_gray(v_in)
  [rows_hr,cols_hr] = v_img.shape
  rows_ctr = rows_hr//2
  cols_ctr = cols_hr//2
  v = v_img[rows_ctr-256:rows_ctr+256,cols_ctr-256:cols_ctr+256]
  epsil_k = np.random.normal(0,sig,v.shape)
  # clip x to be non negative to avoid dark region bleeding issue
  x_img = np.add(v, epsil_k)
  y = construct_nonlinear_model(x_img, sigma, alpha, sigw, gamma=gamma,clip=clip)
  fv = construct_nonlinear_model(v, sigma, alpha, 0, gamma=gamma, clip=clip)
  y_fv_k = np.subtract(y,fv)
  epsil.append(epsil_k) 
  y_fv.append(y_fv_k)
  v_list.append(v)

print(n_samples)
epsil = np.array(epsil)
y_fv = np.array(y_fv)
dataset = {'epsil':epsil,'y_fv':y_fv, 'v':v_list}
if clip:
  dict_name = '/root/datasets/pmap_exp_'+forward_name+'_noisy_small_gauss_clip.dat'
else:
  dict_name = '/root/datasets/pmap_exp_'+forward_name+'_noisy_small_gauss_noclip.dat'
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
