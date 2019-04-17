from scipy.ndimage import convolve
import numpy as np
# function to construct forward model for plug and play super resolution problem
# z: input ground truth image
# h: anti-aliasing filter
# K: down-sampling rate
# sigw: std deviation for AWGN W
# output: y = SHz+W
def construct_forward_model(z, K, h, sigw):
  [rows_hr,cols_hr] = np.shape(z)
  y = convolve(z,h,mode='wrap')
  y = y[::K,::K] # downsample z by taking every Kth pixel
  np.random.seed(0)
  gauss = np.random.normal(0,1,np.shape(y))
  y = np.clip(y+sigw*gauss,0,1)
  return y
 



