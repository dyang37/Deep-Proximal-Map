import numpy as np
import sys,os,argparse
sys.path.append(os.path.join(os.getcwd(), "util"))
from skimage.io import imread, imsave
from math import sqrt
from skimage.measure import compare_psnr
from sr_util import gauss2D, windowed_sinc, avg_filt
from construct_forward_model import construct_forward_model
from icd_simulation import icd_simulation
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
sigw = 10./255. # noise level
#siglam = sigw
#lambd = 1./(siglam*siglam)
lambd=50.
gamma = 1
beta = 1
max_itr = 40

print("Using ",optim_dict[optim_method],"as optimization method for forward model inversion...")
################### Data Proe-processing
fig_in = 'bowl'
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


################## Forward model construction
# Your filter design goes HERE
h1 = gauss2D((33,33),1)
h2 = windowed_sinc(K)
h3=avg_filt(9)
h_list=[h1,h2,h3]
#io.savemat('h_data.mat', mdict={'h': h})
# call function to construct forward model: y=SH+W
h_name_list=['gauss','sinc','avg_filt']
for h,filt_choice in zip(h_list,h_name_list):
  print(filt_choice)
  y = construct_forward_model(z, K, h, 0)
  # save image
  figname = str(K)+'_SR_noisy_input_'+filt_choice+'.png'
  fig_fullpath = os.path.join(os.getcwd(),figname)
  imsave(fig_fullpath, y)
  ################## Plug and play ADMM iterative reconstruction
  icd_simulation(z,y,h,sigw,lambd,K,filt_choice)
