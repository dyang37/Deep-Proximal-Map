import numpy as np
import sys,os,argparse
sys.path.append(os.path.join(os.getcwd(), "util"))
from skimage.io import imread, imsave
from math import sqrt
from skimage.measure import compare_psnr
from sr_util import gauss2D, windowed_sinc, avg_filt
from construct_forward_model import construct_nonlinear_model
from plug_and_play_nonlinear import plug_and_play_nonlinear
import matplotlib.pyplot as plt
import scipy.io as io

################## parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', action='store', dest='optim', help='proximal map update choce. 0: fourier decomposition; 1: ICD',type=int, default=1)
args = parser.parse_args()
optim_method = args.optim #0: Stanley's closed form solution 1: icd update
optim_dict = {0:"deep proximal map", 1:"icd update"}


################### hyperparameters
sigw = 0.1 # noise level
sigma = 10
alpha = 0.5
gamma = 2

print("Using ",optim_dict[optim_method],"as optimization method for forward model inversion...")
################### Data Proe-processing
fig_in = 'test_gray'
z = np.array(imread('./data/'+fig_in+'.png'), dtype=np.float32) / 255.0
filt_choice = 'nonlinear'
print("filter choice: ",filt_choice)
# y = Gz. We deliberately make awgn=0 for the purpose of experiments
y = construct_nonlinear_model(z, sigma, alpha, sigw, gamma=gamma)
  # save image
figname = 'pnp_input_'+filt_choice+'.png'
fig_fullpath = os.path.join(os.getcwd(),figname)
imsave(fig_fullpath, y)
################## Plug and play ADMM iterative reconstruction
x = plug_and_play_nonlinear(y,sigma,alpha,sigw,gamma)
