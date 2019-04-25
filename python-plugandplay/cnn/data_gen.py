import os,sys
sys.path.append(os.path.join(os.getcwd(), "../util"))
import numpy as np
from construct_forward_model import construct_forward_model
from sr_util import gauss2D, windowed_sinc, avg_filt
import pickle
sig = 10./255.
sigw = 10./255.
rows_hr = 372
cols_hr = 500
v = np.random.rand(100000,rows_hr,cols_hr)
x = []
y = []
fv = []
# define forward model and construct y=f(x)+w
K = 4   # downsampling factor
h = windowed_sinc(K)
print("... Forward project data through forward model ...")
for v_sample in v:
  x_sample = np.random.normal(v_sample,sig)
  x.append(x_sample)
  y.append(construct_forward_model(x_sample, K, h, sigw))
  fv.append(construct_forward_model(v_sample, K, h, 0))
print("... Exporting dataset ...")
x=np.array(x)
y=np.array(y)
fv=np.array(fv)
dataset = {"x":x.reshape(x.shape+(1,)),"v":v.reshape(v.shape+(1,)),"y":y.reshape(y.shape+(1,)),"fv":fv.reshape(fv.shape+(1,))}
fd = open('pseudo-proximal-map-dict-large.dat','wb')
pickle.dump(dataset,fd)
print("Done.")
