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
from inpaint_util import shepard_init,generateAmatrix,DataTermOptInPaint
from skimage.io import imread
import numpy as np
from math import sqrt
from skimage.restoration import denoise_tv_chambolle as denoiser_tv
from skimage.measure import compare_psnr
from keras.models import  model_from_json
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
  # optional tuning for nlm denoiser. Currently the code is using default patch setting from skimage library
  patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=False)
else:
  raise Exception('Error: unknown denoiser.')

# hyperparameters
noise_level = 10./255.;
rho = 1.
gamma = 1.
max_itr = 30
tol = 10**-4
lambd = 1./20.

# data pre-processing: apply mask and add AWGN
z = np.array(imread('./data/shoes-hr-gray.png'), dtype=np.float32) / 255.0
print('input image size: ',np.shape(z))
[row,col]=np.shape(z)
N=row*col
np.random.seed(0)
mask = np.random.rand(row,col)
mask=mask>=0.8
gauss = np.random.normal(0,1,(row,col))
z_mask=z*mask
y=np.clip(z_mask+noise_level*gauss,0,1)

# ADMM initialization
v = np.random.rand(row,col)

x = v
u = np.random.rand(row,col)
#u = shepard_init(y, mask, 10) 
residual = float("inf")
itr = 0

Amatrix = np.transpose(generateAmatrix(mask, 0))
# ADMM recursive update
print('itr      residual          mean-sqr-error')
while (residual > tol) and (itr < max_itr):
  mse = (1/sqrt(N))*(sqrt(sum(sum((x-z)**2))))
  x_old = copy.copy(x)
  v_old = v
  u_old = u
  # inversion
  xtilde = v-u
  # icd update
  if itr == 0:
    icd_niter = 10
  else:
    icd_niter = 1
  x = DataTermOptInPaint(x, y, icd_niter, u, v, lambd, Amatrix)
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
  #print(itr,' ',residualx,' ', residualv,'  ', residualu)
  print(itr,' ',residual,'  ',mse)
# end of ADMM recursive update
psnr = compare_psnr(z, x)
print('PSNR of restored image: ',psnr)

plt.figure()
plt.subplot(131)
plt.imshow(z,interpolation='nearest',cmap='gray')
plt.title('original image')
plt.subplot(132)
plt.imshow(y,interpolation='nearest',cmap='gray')
plt.title('noisy image')
plt.subplot(133)
plt.imshow(x,interpolation='nearest',cmap='gray')
plt.title('reconstructed image')
figname = 'inpainting'+imgext
fig_fullpath = os.path.join(os.getcwd(),figname)
plt.savefig(fig_fullpath)
