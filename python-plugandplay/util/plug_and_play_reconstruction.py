import numpy as np
import sys
import os
from math import sqrt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "./denoisers/DnCNN"))
from keras.models import  model_from_json
from scipy.misc import imresize
from dncnn import cnn_denoiser
from skimage.restoration import denoise_tv_chambolle as denoiser_tv
from skimage.restoration import denoise_nl_means
from forward_model_optim import forward_model_optim, icd_update
# This function performs ADMM iterative reconstruction for image super resolution problem
# hr_img: ground_truth image. Only used for evaluation purpose
# y: low resolution input image
# h: anti-aliasing filter used in forward model
# sigw: noise std-deviation
# K: down/up sampling factor
# denoiser: choice for image denoising algorithm
#           0: DnCNN
#           1: Total Variation
#           2: Non-local Mean
# return value: x, the reconstructed high resolution image


def plug_and_play_reconstruction(hr_img,y,h,sigw,beta,lambd,gamma,max_itr,K,denoiser, optim_method):
  if denoiser == 0:
    # you can replace model.json file and model.h5 file with any other pre-trained neural network
    model_dir=os.path.join('models',os.getcwd(),'./denoisers/DnCNN')
    # load json and create model
    json_file = open(os.path.join(model_dir,'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join(model_dir,'model.h5'))

  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr*K
  cols_hr = cols_lr*K
  N = rows_hr*cols_hr
  v = imresize(y, [rows_hr, cols_hr])/255.
  x = v
  u = np.zeros((rows_hr, cols_hr))
  residual = float("inf")
  mse_min = float("inf")
  # hyperparameters
  tol = 10**-5
  patience = 10
  # iterative reconstruction
  print('itr      residual          mean-sqr-error')
  itr = 0
  while((residual > tol) or (fluctuate <= patience)) and (itr < max_itr):
    v_old = v
    u_old = u
    x_old = x
    xtilde = v-u
    # forward model optimization step
    x = optimization_wrapper(x,xtilde,y,h,K,lambd,sigw,itr,optim_method)
    # denoising step
    vtilde = x + u
    vtilde = vtilde.clip(min=0,max=1)
    sigma = sqrt(beta/lambd)
    if denoiser == 0:
      v = cnn_denoiser(vtilde, model)
    elif denoiser == 1:
      v = denoiser_tv(vtilde)
    elif denoiser == 2:
      v = denoise_nl_means(vtilde, sigma=sigma)
    else:
      raise Exception('Error: unknown denoiser.')
    # update u
    u = u+(x-v)
    # update lambd
    lambd = lambd*gamma
    # calculate residual
    residualx = (1/sqrt(N))*(sqrt(sum(sum((x-x_old)**2))))
    residualv = (1/sqrt(N))*(sqrt(sum(sum((v-v_old)**2))))
    residualu = (1/sqrt(N))*(sqrt(sum(sum((u-u_old)**2))))
    residual = residualx + residualv + residualu
    itr = itr + 1
    # calculate mse
    mse = (1/sqrt(N))*(sqrt(sum(sum((x-hr_img)**2))))
    if (mse < mse_min):
      fluctuate = 0
      mse_min = mse
    else:
      fluctuate += 1
    print(itr,' ',residual,'  ', mse)
  return x


def optimization_wrapper(x,xtilde,y,h,K,lambd,sigw,itr,optim_method):
  if optim_method == 0:
    x = forward_model_optim(x,xtilde,y,h,K, lambd)
  elif optim_method == 1:
    if itr == 0:
      for _ in range(10):
        x = icd_update(x,xtilde,y,h,K,lambd,sigw)
    else:
      x = icd_update(x,xtilde,y,h,K,lambd,sigw)
  else:
    raise Exception('Error: unknown optimization method.')
  return x
