import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "util"))
from skimage.io import imread, imsave
from math import sqrt
from skimage.measure import compare_psnr
from sr_util import gauss2D # comment this out if you don't want to use Gaussian as anti-aliasing filter
from sr_forward_model import construct_forward_model
from plug_and_play_reconstruction import plug_and_play_reconstruction


denoiser=int(sys.argv[1])
denoiser_dict = {0:"DnCNN",1:"Total Variation",2:"Non-local Mean"}
print("Using ",denoiser_dict[denoiser],"as denoiser for prior model...")

# hyperparameters
K = 4 # downsampling factor
sigw = 10./255. # noise level
rho = 1.
gamma = 0.99
lambd = 0.01

# data pre-processing: convert image to grayscale and truncate image
fig_in = 'shoes-hr-gray'
z_in = np.array(imread('./data/'+fig_in+'.png'), dtype=np.float32) / 255.0
# check if grayscale
[rows_in,cols_in] = np.shape(z_in)[0:2]
rows_lr = rows_in//K
cols_lr = cols_in//K
if (np.shape(z_in).__len__() > 2):
  # convert RGB to grayscale image
  z=np.zeros((rows_in,cols_in))
  for i in range(rows_in):
    for j in range(cols_in):
      r = z_in[i,j,0]
      g = z_in[i,j,1]
      b = z_in[i,j,2]
      z[i,j]=0.2989 * r + 0.5870 * g + 0.1140 * b
else:
  z = z_in
# truncate the image in case that rows_in cannot be devided by K
z = z[0:rows_lr*K, 0:cols_lr*K]
print('input image size: ',np.shape(z))

###### Your filter design goes HERE
h = gauss2D((9,9),1)
# call function to construct forward model: y=SH+W
y = construct_forward_model(z, K, h, sigw)
# save image
figname = str(K)+'_SR_noisy_input.png'
fig_fullpath = os.path.join(os.getcwd(),figname)
imsave(fig_fullpath, y)
# ADMM iterative reconstruction
map_img = plug_and_play_reconstruction(z,y,h,sigw,lambd,rho,gamma,K,denoiser)

# evaluate performance
psnr = compare_psnr(z, map_img)
print('PSNR of restored image: ',psnr)
# save reconstructed image
figname = str(K)+'_SR_output_'+denoiser_dict[denoiser]+'.png'
fig_fullpath = os.path.join(os.getcwd(),figname)
imsave(fig_fullpath, np.clip(map_img,0,1))
