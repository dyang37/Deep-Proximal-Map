import copy
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "./util"))
sys.path.append(os.path.join(os.getcwd(), "./denoisers/DnCNN"))
from dncnn import cnn_denoiser
from skimage.io import imread, imsave
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
else:
  raise Exception('Error: unknown denoiser.')

# hyperparameters
K_list = [2,2] # downsampling factors
# calculate total SR rate
K_tot = 1
for r in K_list:
  K_tot = K_tot*r

print("overall SR rate: ",K_tot)
noise_level = 10./255.;
rho = 1.
gamma = 0.99
max_itr = 300
tol = 10**-4
lambd = 0.01
patience = 5
fig_in = 'shoes-hr-gray'
z_in = np.array(imread('./data/'+fig_in+'.png'), dtype=np.float32) / 255.0
# check if grayscale
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
[rows_in,cols_in] = np.shape(z)
rows_lr = rows_in // K_tot
cols_lr = cols_in // K_tot
z = z[0:rows_lr*K_tot, 0:cols_lr*K_tot]

h_list=[]
z_list=[]
for stage in range(len(K_list)):
  K = K_list[stage]
  print()
  print('================ input stage #',stage,' downsamping, K=',K,'===================')
  [rows_hr,cols_hr] = np.shape(z)
  rows_lr = rows_hr//K
  cols_lr = cols_hr//K
  print('input image size: ',np.shape(z))
  N=rows_hr*cols_hr
  h = gauss2D((9,9),1)
  y = correlate(z,h,mode='wrap')
  y = y[::K,::K] # downsample z by taking every Kth pixel
  np.random.seed(0)
  figname = 'multistage_SR_input_stage_'+str(stage)+'.png'
  fig_fullpath = os.path.join(os.getcwd(),figname)
  imsave(fig_fullpath, y)
  # input of next stage is output image of previous stage
  z_list.append(z)
  # input of next stage is output of prev stage
  z = copy.copy(y)
  h_list.append(h)
# add noise at the last input stage

gauss = np.random.normal(0,1,np.shape(y))
y = np.clip(y+noise_level*gauss,0,1)
imsave(fig_fullpath, y)

print(" ============== Downsampling all stages completed =============")
for stage in range(len(K_list)):
  print('================ output stage #',stage,' restoring, K=',K,'===================')
  up_stage = len(K_list)-1-stage
  h = h_list[up_stage]
  z = z_list[up_stage]
  K = K_list[up_stage]
  # ADMM initialization
  [rows_lr, cols_lr] = np.shape(y)
  rows_hr = rows_lr * K
  cols_hr = cols_lr*K
  v = imresize(y,[rows_hr,cols_hr])/255.
  x = copy.copy(v)
  print('upsamping rate: ',str(K),'    input size: ',np.shape(y),'    output size: ',np.shape(z))
  u = np.zeros(np.shape(v))
  residual = float("inf")
  mse_min = float("inf")
  fluctuate = 0
  GGt = constructGGt(h,K,rows_hr, cols_hr)
  Gty = construct_Gt(y,h,K)
  # ADMM recursive update
  print('itr      residual          mean-sqr-error')
  itr = 0
  while ((residual > tol) or (fluctuate <= patience)) and (itr < max_itr) :
    v_old = copy.copy(v)
    u_old = copy.copy(u)
    x_old = copy.copy(x)
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
    mse = (1/sqrt(N))*(sqrt(sum(sum((x-z)**2))))
    if (mse < mse_min):
      fluctuate = 0
      mse_min = mse
    else:
      fluctuate += 1
    print(itr,' ',residual,'  ', mse)
    # end of ADMM recursive update
  # update input of next stage to be the output restored image of previous stage
  y = copy.copy(x)
  #end of current stage
  # statistics and image postprocessing for current stage
  psnr = compare_psnr(z, x)
  print('PSNR of restored image: ',psnr)

  figname = 'multistage_SR_output_upstage_'+str(up_stage)+'_rate_'+str(K)+imgext
  fig_fullpath = os.path.join(os.getcwd(),figname)
  imsave(fig_fullpath, np.clip(x,0,1))
figname = 'multistage_SR_output_final'+imgext
fig_fullpath = os.path.join(os.getcwd(),figname)
imsave(fig_fullpath, np.clip(x,0,1))

print(" ============== SR all stages completed =============")
