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

# calculate x = argmin(1/sigw^2*|SHx-y|^2 + beta/2*|x-v|^2)
def forward_model_optim(x,xtilde,y,h,K, lambd):
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  GGt = constructGGt(h,K,rows_hr, cols_hr)
  Gty = construct_Gt(y,h,K)
  rhs = Gty + lambd*xtilde
  G = construct_G(rhs, h, K)
  Gt = construct_Gt(np.abs(ifft2(fft2(G)/(GGt+lambd))),h,K)
  map_img = (rhs - Gt)/lambd
  return map_img

def icd_update(x,xtilde,y,h,K,lambd,sigw):
  [rows_hr, cols_hr] = np.shape(x)
  map_img = copy.deepcopy(x)
  Hx = correlate(x,h,mode='wrap')
  e = Hx[::K,::K]-y
  with tf.device('/device:GPU:2'):
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
      map_img[i,j] += alpha
      # update error image
      e += alpha*Gs
  sess.close()
  return map_img.clip(min=0, max=1)
