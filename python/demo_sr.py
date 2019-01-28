import copy
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "./util"))
sys.path.append(os.path.join(os.getcwd(), "./denoisers/DnCNN"))
import dncnn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
from dncnn import cnn_denoiser
from skimage.io import imread
import numpy as np
from math import sqrt
from skimage.restoration import denoise_tv_chambolle as denoiser_tv
from skimage.measure import compare_psnr
from PIL import Image
#import ADMM_SR as admm
from keras.models import  model_from_json
from sr_util import gauss2D, construct_G, construct_Gt, constructGGt
from scipy.ndimage import correlate
from scipy.misc import imresize
from numpy.fft import fft2, ifft2
from skimage.restoration import denoise_nl_means
denoiser=int(sys.argv[1])
if denoiser == 0:
  # loading pre-trained cnn model...
  print('using neural network denoiser...')
  imgext = '_cnn.png'
  model_dir=os.path.join('models',os.getcwd(),'./denoisers/DnCNN')
  # load json and create model
  json_file = open(os.path.join(model_dir,'model.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights(os.path.join(model_dir,'model.h5'))
elif denoiser == 1:
  print('using total variation denoiser...')
  imgext = '_tv.png'
elif denoiser == 2:
  print('using nlm denoiser...')
  imgext = '_nlm.png'
  patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False)
else:
  raise Exception('Error: unknown denoiser.')

# hyperparameters
K = 2 # downsampling factor
noise_level = 10./255.;
rho = 1.
gamma = 0.99
max_itr = 300
tol = 10**-4
lambd = 0.01

# data pre-processing: apply mask and add AWGN
z_in = np.array(imread('./data/couple512.png'), dtype=np.float32) / 255.0
# check if grayscale
[rows_hr,cols_hr] = np.shape(z_in)[0:2]
if (np.shape(z_in).__len__() > 2):
  # convert RGB to grayscale image
  z=np.zeros((rows_hr,cols_hr))
  for i in range(rows_hr):
    for j in range(cols_hr):
      r = z_in[i,j,0]
      g = z_in[i,j,1]
      b = z_in[i,j,2]
      z[i,j]=0.2989 * r + 0.5870 * g + 0.1140 * b
else:
  z = z_in
print('input image size: ',np.shape(z))
N=rows_hr*cols_hr
rows_lr = rows_hr//K
cols_lr = cols_hr//K
'''
y = np.zeros((rows_lr, cols_lr))
for i in range(rows_lr):
  for j in range(cols_lr):
    y[i,j] = np.mean(z[K*i:K*(i+1),K*j:K*(j+1)])
'''
h = gauss2D((9,9),1)
y = correlate(z,h,mode='wrap')
y = y[::K,::K] # downsample z by taking every Kth pixel
np.random.seed(0)
gauss = np.random.normal(0,1,np.shape(y))
y = np.clip(y+noise_level*gauss,0,1)
plt.figure()
plt.imshow(y,cmap='gray')
plt.title('downsampled noisy image')
figname = repr(K)+'_'+str(max_itr)+'_SR_noisy_input'+imgext
fig_fullpath = os.path.join(os.getcwd(),figname)
plt.savefig(fig_fullpath)
# ADMM initialization
v = imresize(y,[rows_hr,cols_hr])/255.
x = v
u = np.zeros(np.shape(v))
residual = float("inf")
GGt = constructGGt(h,K,rows_hr, cols_hr)
Gty = construct_Gt(y,h,K)
# ADMM recursive update
itr = 0
while (residual > tol) and (itr < max_itr):
  mse = (1/sqrt(N))*(sqrt(sum(sum((x-z)**2))))
  v_old = v
  u_old = u
  x_old = x
  # inversion
  xtilde = v-u
  rhs = Gty + rho*xtilde
  G = construct_G(rhs, h, K)
  Gt = construct_Gt(np.abs(ifft2(fft2(G)/(GGt+rho))),h,K)
  x = (rhs - Gt)/rho
  
  # denoising
  vtilde=x+u
  vtilde = vtilde.clip(min=0,max=1)
  sigma = sqrt(lambd/rho)
  if denoiser == 0:
    v = cnn_denoiser(vtilde, model)
  elif denoiser == 1:
    v = denoiser_tv(vtilde)
  else:
    v = denoise_nl_means(vtilde, sigma=sigma)
  # update u
  u = u+(x-v)
  # update rho
  rho=rho*gamma
  residualx = (1/sqrt(N))*(sqrt(sum(sum((x-x_old)**2))))
  residualv = (1/sqrt(N))*(sqrt(sum(sum((v-v_old)**2))))
  residualu = (1/sqrt(N))*(sqrt(sum(sum((u-u_old)**2))))
  residual = residualx + residualv + residualu
  itr = itr + 1
  print(itr,' ',residual,'  ', mse)
  # end of ADMM recursive update


psnr = compare_psnr(z, x)
print('PSNR of restored image: ',psnr)

plt.figure()
plt.subplot(121)
plt.imshow(z,cmap='gray')
plt.title('original image')
plt.subplot(122)
plt.imshow(x,cmap='gray')
plt.title('reconstructed image')
figname = repr(K)+'_SR_'+repr(lambd)+imgext
fig_fullpath = os.path.join(os.getcwd(),figname)
plt.savefig(fig_fullpath)
