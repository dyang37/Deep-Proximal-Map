import numpy as np
import sys,os
sys.path.append(os.path.join(os.getcwd(), "../util"))
from skimage.io import imread, imsave
from math import sqrt
from skimage.measure import compare_psnr
from sr_util import gauss2D, windowed_sinc, avg_filt
from construct_forward_model import construct_nonlinear_model
from mace import mace
import matplotlib.pyplot as plt
import scipy.io as io


################### hyperparameters
clip = False
sigw = 0.1 # noise level
sigma_g = 10
alpha = 0.5
gamma = 2
sig = 0.05
beta = 0.2
p = 3

################### Data Proe-processing
fig_in = 'test_gray'
z = np.array(imread('../'+fig_in+'.png'), dtype=np.float32) / 255.0
filt_choice = 'nonlinear'
print("filter choice: ",filt_choice)
# y = Gz. We deliberately make awgn=0 for the purpose of experiments
y = construct_nonlinear_model(z, sigma_g, alpha, sigw, gamma=gamma, clip=clip)
  # save image
figname = 'pnp_input_'+filt_choice+'.png'
fig_fullpath = os.path.join(os.getcwd(),figname)
imsave(fig_fullpath, y)
################## Plug and play ADMM iterative reconstruction
x = mace(p,z,y,sigma_g,alpha,beta,sigw,sig,gamma,clip,rho=0.2)

