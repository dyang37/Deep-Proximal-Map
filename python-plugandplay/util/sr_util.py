import numpy as np
import copy
from scipy.ndimage import correlate
from scipy.signal import convolve2d
from numpy.fft import fft2

def construct_Gs(h,rows,cols,i,j,K):
  [rows_h, cols_h] = np.shape(h)
  h_img = np.zeros((rows,cols))
  ctr_i_h = rows_h // 2
  ctr_j_h = cols_h // 2
  if (i-ctr_i_h >= 0) and (i-ctr_i_h+rows_h < rows) and (j-ctr_j_h>=0) and (j-ctr_j_h+cols_h < cols):
    h_img[i-ctr_i_h:i+ctr_i_h+1, j-ctr_j_h:j+ctr_j_h+1] = h
  else:
    for di in range(-ctr_i_h,-ctr_i_h+rows_h):
      for dj in range(-ctr_j_h,-ctr_j_h+cols_h):
        h_img[(di+i)%rows,(dj+j)%cols] = h[di+ctr_i_h,dj+ctr_j_h]
  return h_img[::K,::K]

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
  yc = rows_hth//2
  xc = cols_hth//2
  L = int(np.ceil(rows_hth/K))  # width of the new filter
  g = np.zeros((L,L))
  for i in range(-(L//2),L//2+1):
    for j in range(-(L//2),L//2+1):
      g[i+L//2,j+L//2] = hth[yc+K*i-1,xc+K*j-1]
  GGt = np.abs(fft2(g,[rows//K,cols//K]))
  return GGt

# 2d gaussian mask filtered by a hamming window
def gauss2D(shape=(15,15),sigma=0.5):
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  w_0 = np.hamming(shape[0])
  w_1 = np.hamming(shape[1])
  w = w_0*w_1
  h = np.multiply(h,w)
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

def gauss2D_nowindow(shape=(15,15),sigma=0.5):
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

def windowed_sinc(K):
  print('------- Using sinc filter as forward model -------')
  p = 8*K+1
  n = np.arange(p)
  hsinc = np.sinc((n - (p - 1) / 2.)/K)/K
  w = np.hamming(p)
  h_1d = hsinc * w
  h_1d = h_1d / np.sum(h_1d)
  h = np.zeros((p,p))
  for i in range(p):
    for j in range(p):
      h[i,j] = h_1d[i]*h_1d[j]
  return h

def avg_filt(p):
  print('------- Using avg filter as forward model -------')
  h=np.ones((p,p))
  h=h/np.sum(h[:])
  return h

