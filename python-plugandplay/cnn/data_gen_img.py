import os,sys
sys.path.append(os.path.join(os.getcwd(), "../util"))
import numpy as np
from construct_forward_model import construct_forward_model
from sr_util import gauss2D, windowed_sinc, avg_filt
import pickle

def to_gray(x_in, K):
  [rows_in,cols_in] = np.shape(x_in)[0:2]
  rows_out = rows_in//K*K
  cols_out = cols_in//K*K
  x = np.zeros(rows_out, cols_out)
  for i in range(rows_out):
    for j in range(cols_out):
      r = x_in[i,j,0]
      g = x_in[i,j,1]
      b = x_in[i,j,2]
      x[i,j]=0.2989 * r + 0.5870 * g + 0.1140 * b
  return x

sig = 60./255.
sigw = 60./255.
rows_hr = 372
cols_hr = 500
K = 4   # downsampling factor
filt_name = 'sinc'
print('filter name:',filt_name)
h = windowed_sinc(K)
#h = avg_filt(9)
#h = gauss2D((33,33),1)
# define forward model and construct y=f(x)+w
database_dir = os.path.abspath('/root/ML/datasets/pmap')
for img_name in os.walk(directory):
  x_in = np.array(imread(img_name), dtype=np.float32) / 255.0
  x = to_gray(x_in, K)
  v = np.random.normal(x,sig)
  y = construct_forward_model(x, K, h, sigw)
  fv = construct_forward_model(v, K, h, 0)
  print("... Exporting dataset ...")
  dataset = {"x":x,"v":v_dict,"y":y,"fv":fv}
  fd = open(dict_name,'wb')
  pickle.dump(dataset,fd)
  fd.close()
  print('dictionary ',n_dict,'exported.')

print("Done.")
