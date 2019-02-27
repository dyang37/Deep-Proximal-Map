import numpy as np
import copy
from scipy.ndimage import correlate
from scipy.signal import convolve2d
from numpy.fft import fft2

def construct_G(x,h,K):
  tmp = correlate(x,h,mode='wrap')
  y = tmp[::K,::K]
  return y

def construct_Gt(x,h,K):
  [rows,cols] = np.shape(x)
  tmp = np.zeros((rows*K,cols*K))
  tmp[::K,::K] = x[:,:]
  y = correlate(tmp,h,mode='wrap')
  return y

# eigen decomposition for super-resolution
# cols and rows should be divisible by K
def constructGGt(h,K,rows,cols):
  hth = convolve2d(h,np.rot90(h,2),'full')
  [rows_hth,cols_hth] = np.shape(hth)
  # center coordinates
  yc = int(np.ceil(rows_hth/2.))
  xc = int(np.ceil(cols_hth/2.))
  L = int(np.ceil(rows_hth/K))   # width of the new filter
  g = np.zeros((L,L))
  for i in range(-(L//2),L//2+1):
    for j in range(-(L//2),L//2+1):
      g[i+L//2,j+L//2] = hth[yc+K*i-1,xc+K*j-1]
  GGt = np.abs(fft2(g,[rows//K,cols//K]))
  return GGt

# 2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])   
def gauss2D(shape=(3,3),sigma=0.5):
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h
