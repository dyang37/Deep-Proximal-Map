import numpy as np
import sys,os
sys.path.append(os.path.join(os.getcwd(), "../util"))
from skimage.io import imread, imsave
from math import sqrt
from skimage.measure import compare_psnr
from sr_util import gauss2D, windowed_sinc, avg_filt
from construct_forward_model import construct_nonlinear_model
from plug_and_play_nonlinear import plug_and_play_nonlinear
import matplotlib.pyplot as plt
import scipy.io as io


################### hyperparameters
clip = False
sigw = 0.1 # noise level
sigma = 10
alpha = 0.5
gamma = 2

################### Data Proe-processing
fig_in = 'test_gray'
z = np.array(imread('../'+fig_in+'.png'), dtype=np.float32) / 255.0
filt_choice = 'nonlinear'
print("filter choice: ",filt_choice)
# y = Gz. We deliberately make awgn=0 for the purpose of experiments
y = construct_nonlinear_model(z, sigma, alpha, sigw, gamma=gamma, clip=clip)
  # save image
figname = 'pnp_input_'+filt_choice+'.png'
fig_fullpath = os.path.join(os.getcwd(),figname)
imsave(fig_fullpath, y)
################## Plug and play ADMM iterative reconstruction
x = plug_and_play_nonlinear(z,y,sigma,alpha,sigw,gamma,clip)
