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
from sr_inversion import sr_inversion
#import ADMM_SR as admm
from keras.models import  model_from_json
from sr_util import gauss2D
from scipy.ndimage import convolve


denoiser=int(sys.argv[1])
if denoiser == 0:
  # loading pre-trained cnn model...
  print('using neural network denoiser...')
  model_dir=os.path.join('models',os.getcwd(),'./denoisers/DnCNN')
  # load json and create model
  json_file = open(os.path.join(model_dir,'model.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  # load weights into new model
  model.load_weights(os.path.join(model_dir,'model.h5'))
else:
  print('using total variation denoiser...')

# hyperparameters
noise_level = 10./255.;
rho = 1.
gamma = 1.
max_itr = 20
tol = 10**-4
lambd = 1./20.
var_estm = noise_level  # we are cheating here :(

# data pre-processing: apply mask and add AWGN
z = np.array(imread('./data/Couple512.png'), dtype=np.float32) / 255.0
print('input image size: ',np.shape(z))
R = 2   # downsampling factor
[rows_hr,cols_hr] = np.shape(z)
N=rows_hr*cols_hr
rows_lr = rows_hr//R
cols_lr = cols_hr//R
'''
y = np.zeros((rows_lr, cols_lr))
for i in range(rows_lr):
  for j in range(cols_lr):
    y[i,j] = np.mean(z[R*i:R*(i+1),R*j:R*(j+1)])
'''
h = gauss2D((9,9),1)
y = convolve(z,h,mode='wrap')
y = y[::R,::R] # downsample z by taking every Rth pixel
np.random.seed(0)
gauss = np.random.normal(0,1,np.shape(y))
y = np.clip(y+noise_level*gauss,0,1)
plt.imshow(y,'gray')
#plt.show()
# ADMM initialization
v = np.random.rand(rows_hr,cols_hr)
x = v
u = np.random.rand(rows_hr,cols_hr)
residual = float("inf")

# ADMM recursive update
mu = np.zeros((rows_lr,cols_lr))
itr = 0
while (residual > tol) and (itr < max_itr):
  err = (1/sqrt(N))*(sqrt(sum(sum((x-z)**2))))
  x_old = copy.copy(x)
  v_old = v
  u_old = u
  # inversion with ICD
  xtilde = v-u
  print(np.shape(xtilde))
  for i in range(rows_lr):
    for j in range(cols_lr):
      mu[i,j] = np.mean(x_old[i*R:(i+1)*R, j*R:(j+1)*R])
  if itr == 0:
    icd_niter = 10
  else:
    icd_niter = 1
  x = sr_inversion(mu,y,xtilde,R,icd_niter, lambd)
  # denoising
  vtilde=x+u
  vtilde = vtilde.clip(min=0,max=1)
  sigma = sqrt(lambd/rho)
  if denoiser == 0:
    v = cnn_denoiser(vtilde, model)
  else:
    v = denoiser_tv(vtilde)
  # update u
  u = u+(x-v)
  # update rho
  rho=rho*gamma
  residualx = (1/sqrt(N))*(sqrt(sum(sum((x-x_old)**2))))
  residualv = (1/sqrt(N))*(sqrt(sum(sum((v-v_old)**2))))
  residualu = (1/sqrt(N))*(sqrt(sum(sum((u-u_old)**2))))
  residual = residualx + residualv + residualu
  itr = itr + 1
  print(itr,' ',residualx,' ', residualv,'  ', residualu,'  ', err)
  # end of ADMM recursive update


psnr = compare_psnr(z, x)
print('PSNR of restored image: ',psnr)

plt.figure()
plt.imshow(z,interpolation='nearest',cmap='gray')
plt.title('original image')
plt.figure()
plt.imshow(y,interpolation='nearest',cmap='gray')
plt.title('noisy image')
plt.figure()
plt.imshow(x,interpolation='nearest',cmap='gray')
plt.title('reconstructed image')
plt.show()

