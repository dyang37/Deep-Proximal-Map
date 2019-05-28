from construct_forward_model import construct_forward_model
from sr_util import windowed_sinc, gauss2D, avg_filt
import numpy as np
from skimage.io import imread

K = 4
sigw = 0.2
fig_in = 'test_gray'
#z = np.array(imread('../data/'+fig_in+'.png'), dtype=np.float32) / 255.0
z = np.zeros((256,256))
h = windowed_sinc(K)
filt_choice = 'sinc'
w = np.random.normal(0,sigw,z.shape)
zw = np.add(z,w)
fz = construct_forward_model(z,K,h,0)
fw = construct_forward_model(w,K,h,0)
lhs = construct_forward_model(zw,K,h,0)
rhs = np.add(fz,fw) 
print('f(0) = ',fz)

