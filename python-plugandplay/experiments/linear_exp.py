import numpy as np
import sys,os,argparse
sys.path.append(os.path.join(os.getcwd(), "../util"))
from skimage.io import imread, imsave
from math import sqrt
from skimage.measure import compare_psnr
from sr_util import gauss2D, windowed_sinc, avg_filt
from construct_forward_model import construct_forward_model
from ml_estim import ml_estimate 
import matplotlib.pyplot as plt
import scipy.io as io

################## parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', action='store', dest='denoiser', help='Denoiser choice. 0: DnCNN; 1: TV; 2: nlm',type=int, default=0)
parser.add_argument('-f', action='store', dest='optim', help='proximal map update choce. 0: fourier decomposition; 1: ICD',type=int, default=1)
args = parser.parse_args()
denoiser = args.denoiser
optim_method = args.optim #0: Stanley's closed form solution 1: icd update
denoiser_dict = {0:"DnCNN",1:"Total Variation",2:"Non-local Mean"}
optim_dict = {0:"fft closed form approximation", 1:"icd update"}
print("Using ",denoiser_dict[denoiser],"as denoiser for prior model...")


################### hyperparameters
K = 4 # downsampling factor
sigw = 0.05
sig = 0.2

print("Using ",optim_dict[optim_method],"as optimization method for forward model inversion...")
################### Data Proe-processing
fig_in = 'test_gray'
z_in = np.array(imread('../'+fig_in+'.png'), dtype=np.float32) / 255.0
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


################## Forward model construction
# Your filter design goes HERE
h = windowed_sinc(K)
#h = gauss2D((33,33),1)
#h = avg_filt(9)
filt_choice = 'sinc'
print("filter choice: ",filt_choice)
# y = Gz. We deliberately make awgn=0 for the purpose of experiments
y = construct_forward_model(z, K, h, sigw)
################## recursive Pseudo ML reconstruction
ml_estimate(y,h,sigw,sig,K,filt_choice)
