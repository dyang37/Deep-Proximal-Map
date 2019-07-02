import numpy as np
from scipy.ndimage import convolve
from construct_forward_model import construct_nonlinear_model
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

# sanity check for gamma=1 case
def grad_nonlinear(x,y,sigma,alpha,sigw,gamma,clip=True):
  Ax = construct_nonlinear_model(x,sigma,alpha,0,gamma=gamma,clip=clip)
  y_Ax = y-Ax
  y_lin = y**gamma
  x_lin = x**gamma
  dy_dylin = y_lin**(1./gamma-1)
  dxlin_dx = x**(gamma-1)
  filt_input = dxlin_dx*y_Ax
  g = gauss2D((15,15),sigma)
  filt_output = alpha*convolve(filt_input,g,mode='wrap')+(1-alpha)*filt_input
  grad_fx = -filt_output*dy_dylin / (sigw*sigw)
  print('range of grad{f(x)}: [',grad_fx.min(),', ',grad_fx.max(),']')
  return grad_fx

def grad_nonlinear_tf(x,y,sigma_g,alpha,sigw,gamma,clip=True):
  [rows,cols] = np.shape(x)
  x_tf = tf.convert_to_tensor(x,dtype=tf.float32)
  y_tf = tf.convert_to_tensor(y,dtype=tf.float32)
  gamma = tf.constant(gamma,dtype=tf.float32)
  alpha = tf.constant(alpha,dtype=tf.float32)
  sigw = tf.constant(sigw,dtype=tf.float32)
  x_lin = x_tf**gamma
  #manually creat wrapping boundary condition
  g = tf.convert_to_tensor(gauss2D((15,15),sigma_g), dtype=tf.float32)
  x_lin_4d = tf.reshape(x_lin,[1,rows,cols,1])
  g_4d = tf.reshape(g,[15,15,1,1])
  filter_output = tf.reshape(tf.nn.convolution(x_lin_4d,g_4d,padding='SAME'),[rows,cols])
  Ax_tf = (alpha*filter_output + (1-alpha)*x_lin)**(1./gamma)
  y_Ax = y_tf-Ax_tf
  fx_tf = tf.reduce_sum(tf.multiply(y_Ax,y_Ax))/(sigw*sigw)
  grad_fx_tf = tf.gradients(fx_tf,x_tf)
  with tf.Session() as sess:
    grad_fx = sess.run(grad_fx_tf)
  return np.array(grad_fx[0])

