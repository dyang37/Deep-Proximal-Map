import numpy as np
from scipy.ndimage import convolve
from forward_model import camera_model
from sr_util import gauss2D
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt

def grad_f(K,h,noise,sigw):
  [rows_lr,cols_lr]=np.shape(noise)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  grad = np.zeros((rows_hr,cols_hr))
  grad[::K,::K] = noise
  grad = convolve(grad,h,mode='wrap')
  grad = 1./(sigw*sigw)*grad
  return grad

def grad_nonlinear_tf(x,y,sigma_g,alpha,sigw,gamma,clip=True):
  [rows,cols] = np.shape(x)
  lengthH = 15
  x_tf = tf.convert_to_tensor(x,dtype=tf.float32)
  y_tf = tf.convert_to_tensor(y,dtype=tf.float32)
  gamma = tf.constant(gamma,dtype=tf.float32)
  alpha = tf.constant(alpha,dtype=tf.float32)
  sigw = tf.constant(sigw,dtype=tf.float32)
  x_lin = x_tf**gamma
  #manually creat wrapping boundary condition
  g = tf.convert_to_tensor(gauss2D((lengthH,lengthH),sigma_g), dtype=tf.float32)
  x_lin_4d = tf.reshape(x_lin,[1,rows,cols,1])
  g_4d = tf.reshape(g,[lengthH,lengthH,1,1])
  filter_output = tf.reshape(tf.nn.convolution(x_lin_4d,g_4d,padding='SAME'),[rows,cols])
  Ax_tf = (alpha*filter_output + (1-alpha)*x_lin)**(1./gamma)
  y_Ax = y_tf-Ax_tf
  fx_tf = tf.reduce_sum(tf.multiply(y_Ax,y_Ax))/(2*sigw*sigw)
  grad_fx_tf = tf.gradients(fx_tf,x_tf)
  with tf.Session() as sess:
    grad_fx = sess.run(grad_fx_tf)
  return np.array(grad_fx[0])

