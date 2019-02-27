import numpy as np
from numpy.fft import fft2, ifft2
from sr_util import constructGGt, construct_Gt, construct_G

# calculate x = argmin(1/sigw^2*|SHx-y|^2 + lambd/2*|x-v|^2)
def forward_model_optim(x,xtilde,y,h,sigw,K, rho):
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

