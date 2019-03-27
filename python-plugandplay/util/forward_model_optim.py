import numpy as np
from scipy.ndimage import correlate
from numpy.fft import fft2, ifft2
from sr_util import construct_Gs, constructGGt, construct_Gt, construct_G
import copy
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import timeit

# calculate x = argmin(1/sigw^2*|SHx-y|^2 + beta/2*|x-v|^2)
def forward_model_optim(x,xtilde,y,h,K, lambd,sigw):
  rho = 2*lambd*sigw*sigw
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  GGt = constructGGt(h,K,rows_hr, cols_hr)
  Gty = construct_Gt(y,h,K)
  rhs = Gty + rho*xtilde
  G = construct_G(rhs, h, K)
  Gt = construct_Gt(np.abs(ifft2(fft2(G)/(GGt+rho))),h,K)
  map_img = (rhs - Gt)/rho
  return map_img

# calculate x = argmin(1/sigw^2*|SHx-y|^2 + beta/2*|x-v|^2) using icd update
def icd_update(x,xtilde,y,h,K,lambd,sigw):
  [rows_hr, cols_hr] = np.shape(x)
  Hx = correlate(x,h,mode='wrap')
  e = Hx[::K,::K]-y
  with tf.device('/device:GPU:1'):
    e_tensor = tf.placeholder(shape=[rows_hr//K, cols_hr//K],dtype=tf.float64)
    Gs_tensor = tf.placeholder(shape=[rows_hr//K, cols_hr//K],dtype=tf.float64)
    etG_tensor = tf.reduce_sum(tf.multiply(e_tensor,Gs_tensor))
    GtG_tensor = tf.reduce_sum(tf.multiply(Gs_tensor,Gs_tensor))
  sess = tf.Session(config=config)
  for i in range(rows_hr):
    for j in range(cols_hr):
      #Gs = tf.convert_to_tensor(filt_G(h,rows_hr, cols_hr, i, j,K))
      Gs = construct_Gs(h,rows_hr, cols_hr, i, j,K)
      etG, GtG = sess.run([etG_tensor, GtG_tensor], feed_dict={e_tensor:e,Gs_tensor:Gs})
      alpha = (lambd*(xtilde[i,j]-x[i,j]) - etG/(sigw*sigw)) / (lambd + GtG/(sigw*sigw))
      x[i,j] = max(alpha+x[i,j],0)
      # update error image
      e += alpha*Gs
  sess.close()
  return x.clip(min=0, max=1)

def norm_Gs(i,j,h,rows_hr, cols_hr, K):
  [h_row, h_col] = np.shape(h)
  h_row_half = h_row//2
  h_col_half = h_col//2
  row_indices = range(0,rows_hr,K)
  col_indices = range(0,cols_hr,K)
  h_row_indices = np.mod(range(i-h_row_half,i+h_row_half+1), rows_hr)
  h_col_indices = np.mod(range(j-h_col_half,j+h_col_half+1), cols_hr)
  g_row_indices = set(row_indices)&set(h_row_indices)
  g_col_indices = set(col_indices)&set(h_col_indices)
  gs_norm = 0
  for g_i in g_row_indices:
    h_i = (g_i+h_row_half-i) % rows_hr
    for g_j in g_col_indices:
      h_j = (g_j+h_col_half-j) % cols_hr
      gs_norm += h[h_i,h_j]*h[h_i,h_j]
  return gs_norm 
