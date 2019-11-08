from scipy.ndimage import convolve
import numpy as np
from sr_util import gauss2D, gauss2D_nowindow
# function to construct forward model for plug and play super resolution problem
# z: input ground truth image
# h: anti-aliasing filter
# K: down-sampling rate
# sigw: std deviation for AWGN W
# output: y = SHz+W
def downsampling_model(z, K, h, sigw):
  [rows_hr,cols_hr] = np.shape(z)
  y = convolve(z,h,mode='wrap')
  y = y[::K,::K] # downsample z by taking every Kth pixel
  np.random.seed(0)
  gauss = np.random.normal(0,1,np.shape(y))
  y = y+sigw*gauss
  return y
 

def camera_model(z,sigma_g,alpha,sigw, gamma=2.2, clip=True):
  if clip:
    z = np.clip(z,0,None)
  z_linear = z**gamma
  g = gauss2D((15,15),sigma_g)
  #g = gauss2D_nowindow((9,9),sigma_g)
  y_linear = alpha*convolve(z_linear,g,mode='wrap')+(1-alpha)*z_linear
  y = y_linear**(1./gamma)
  # awgn
  gauss = np.random.normal(0,1,np.shape(y))
  y = y+sigw*gauss
  return y

def blockAvgModel(x,L=2):
  (rows,cols) = np.shape(x)
  Ax = np.zeros((rows//L,cols//L))
  for i in range(0,rows//L):
    for j in range(0,cols//L):
      Ax[i,j] = x[L*i:L*(i+1),L*j:L*(j+1)].mean(axis=None)
  return Ax
