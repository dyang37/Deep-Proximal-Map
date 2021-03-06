import numpy as np
import sys,os,argparse
sys.path.append(os.path.join(os.getcwd(), "../util"))
from skimage.io import imread, imsave
from math import sqrt
from forward_model import camera_model
from ml_estim_nonlinear import ml_estimate_nonlinear 
import matplotlib.pyplot as plt
import scipy.io as io

################### hyperparameters
clip = False
sigw = 0.05
sig = 0.05
sigma_g = 10
alpha = 0.5
gamma = 2.
################### Data Proe-processing
fig_in = 'test_gray'
z = np.array(imread('../'+fig_in+'.png'), dtype=np.float32) / 255.0
print('input image size: ',np.shape(z))

################## Forward model construction
y = camera_model(z, sigma_g, alpha, sigw, gamma=gamma, clip=clip)
ml_estimate_nonlinear(y,sigma_g,alpha,sig,sigw,gamma,clip)
