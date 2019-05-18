import os,sys
sys.path.append(os.path.join(os.getcwd(), "../util"))
import numpy as np
from construct_forward_model import construct_forward_model
from sr_util import gauss2D, windowed_sinc, avg_filt
import pickle

n_dicts = 20
n_samples = 2000
sig = 60./255.
sigw = 60./255.
rows_hr = 372
cols_hr = 500
K = 4   # downsampling factor
filt_name = 'sinc'
print('filter name:',filt_name)
h = windowed_sinc(K)
#h = avg_filt(9)
h = gauss2D((33,33),1)
# define forward model and construct y=f(x)+w
for n_dict in range(n_dicts):
  dict_name = '/root/ML/datasets/pseudo-proximal-map-dict-'+str(n_dict)+'-sig60-'+filt_name+'.dat'
  x = []
  y = []
  fv = []
  v_dict = []
  for _ in range(n_samples):
    v_sample = np.random.rand(rows_hr,cols_hr)
    x_sample = np.random.normal(v_sample,sig)
    v_dict.append(v_sample)
    x.append(x_sample)
    y.append(construct_forward_model(x_sample, K, h, sigw))
    fv.append(construct_forward_model(v_sample, K, h, 0))
  print("... Exporting dataset ...")
  x=np.array(x)
  y=np.array(y)
  fv=np.array(fv)
  dataset = {"x":x,"v":v_dict,"y":y,"fv":fv}
  fd = open(dict_name,'wb')
  pickle.dump(dataset,fd)
  fd.close()
  print('dictionary ',n_dict,'exported.')

print("Done.")
