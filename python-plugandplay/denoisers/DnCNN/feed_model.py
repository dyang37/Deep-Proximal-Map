# -*- coding: utf-8 -*-
# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/husqin/DnCNN-keras
# =============================================================================

# run this to test the model

import os
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave

    
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))



def cnn_denoiser(y, model):
  y_ = to_tensor(y)
  x_ = model.predict(y_) # inference
  x = from_tensor(x_)
  return x
        
def pseudo_prox_map(y_fv, model):
  y_fv = y_fv.reshape((1,)+y_fv.shape)
  H = model.predict(y_fv) # inference
  return H[0]

def pseudo_prox_map_nonlinear(y_fv, v, model):
  y_fv = y_fv.reshape((1,)+y_fv.shape)
  v = v.reshape((1,)+v.shape)
  H = model.predict([y_fv,v]) # inference
  return H[0]

