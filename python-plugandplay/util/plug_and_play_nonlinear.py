import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imsave
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.models import  model_from_json
import copy
from dncnn import cnn_denoiser, pseudo_prox_map_nonlinear
from gmrf import gmrf_denoiser
from construct_forward_model import construct_nonlinear_model
from grad import grad_nonlinear_tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from skimage.restoration import denoise_tv_chambolle as denoiser_tv
from math import sqrt

def plug_and_play_nonlinear(p,z,y,sigma_g,alpha,beta,sigw,sig,gamma,clip):
  # output dir parse
  if p == 1:
    print("using tv as prior")
    output_dir = os.path.join(os.getcwd(),'../results/pnp_output/nonlinear/tv')
  elif p == 2:
    print("using gmrf as prior")
    output_dir = os.path.join(os.getcwd(),'../results/pnp_output/nonlinear/gmrf')
  else:
    print("using dncnn as prior")
    output_dir = os.path.join(os.getcwd(),'../results/pnp_output/nonlinear/dncnn')
  output_dir = os.path.join(output_dir, str(sigw))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  # save input y 
  imsave(os.path.join(output_dir,'y_input.png'), np.clip(y,0,1))
  
  # parse denoiser model dir
  denoiser_dir=os.path.join(os.getcwd(),'../denoisers/DnCNN')
  json_file = open(os.path.join(denoiser_dir,'model.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  denoiser_model = model_from_json(loaded_model_json)
  denoiser_model.load_weights(os.path.join(denoiser_dir,'model.h5'))
  
  # parse deep proximal map model dir
  pmap_dir=os.path.join(os.getcwd(),'../cnn/nonlinear_model')
  pmap_model_name = "model_nonlinear_"
  if clip:
    pmap_model_name += "clip_sigw_"
  else:
    pmap_model_name += "noclip_sigw_"
  pmap_model_name += str(sigw)
  json_file = open(os.path.join(pmap_dir, pmap_model_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  pmap_model = model_from_json(loaded_model_json)
  pmap_model.load_weights(os.path.join(pmap_dir,pmap_model_name+'.h5'))
  [rows, cols] = np.shape(y)
  
  # estimate sig_x using original image z
  g = np.array([[1./12., 1./6., 1./12.],[1./6.,0,1./6.],[1./12.,1./6.,1./12.]])
  sigx = 0
  for i in range(rows):
    for j in range(cols):
      for di in range(-1,2):
        for dj in range(-1,2):
          sigx += g[di+1,dj+1] * abs((z[i,j]-z[(i+di)%rows,(j+dj)%cols]))**p
  # divide by 2 because we counted each clique twice
  sigx /= 2.
  sigx = (sigx/(rows*cols))**(1./p)
  print('estimated GGMRF sigma = ',sigx) 
  
  ########## real plug and play algorithm starts HERE
  # initialization  
  x = np.random.rand(rows, cols)
  v = np.random.rand(rows, cols)
  u = np.zeros((rows, cols))
  x_dp = copy.deepcopy(x)
  v_dp = copy.deepcopy(v)
  u_dp = copy.deepcopy(u)
  residual_x = []
  residual_v = []
  admm_cost = []
  admm_cost_dp = []
  # iterative reconstruction
  for itr in range(30):
    print("iteration ",itr)
    # forward step
    x_old = copy.deepcopy(x)
    v_old = copy.deepcopy(v)
    xtilde = np.subtract(v,u)
    xtilde_dp = np.subtract(v_dp,u_dp)
    Axtilde = construct_nonlinear_model(xtilde,sigma_g,alpha,0,gamma=gamma, clip=clip)
    Axtilde_dp = construct_nonlinear_model(xtilde_dp,sigma_g,alpha,0,gamma=gamma, clip=clip)
    H = pseudo_prox_map_nonlinear(np.subtract(y,Axtilde_dp),xtilde_dp,pmap_model)
    grad_f_tf = grad_nonlinear_tf(xtilde,y,sigma_g,alpha,sigw,gamma,clip=clip)
    sig_gradf_tf = -sig*sig*grad_f_tf
    x_dp = np.clip(np.add(xtilde_dp, H), 0, None)
    x = np.clip(np.add(xtilde, sig_gradf_tf), 0, None)
    vtilde = np.add(x,u)
    vtilde_dp = np.add(x_dp,u_dp)
    # denoising step
    sig_n = sqrt(beta)*sig
    if p==1:
      v = denoiser_tv(vtilde, sig_n)
      v_dp = denoiser_tv(vtilde_dp,sig_n)
    elif p==2:
      v = gmrf_denoiser(v,vtilde,sig_n,sigx)
      v_dp = gmrf_denoiser(v_dp,vtilde_dp,sig_n,sigx)
    else:
      v = cnn_denoiser(vtilde, denoiser_model)
      v_dp = cnn_denoiser(vtilde_dp, denoiser_model)

    # calculate admm cost
    Ax = construct_nonlinear_model(x,sigma_g,alpha,0,gamma=gamma,clip=clip)
    ggmrf_sum = 0
    for i in range(rows):
      for j in range(cols):
        for di in range(-1,2):
          for dj in range(-1,2):
            ggmrf_sum += g[di+1,dj+1] * abs(x[i,j]-x[(i+di)%rows,(j+dj)%cols])**p
    # divide by 2 because we counted each clique twice
    ggmrf_sum /= 2.
    cost_prior = ggmrf_sum/(p*sigx**p)
    admm_cost.append(sum(sum((y-Ax)**2))/(2*sigw*sigw) + beta*cost_prior)
    Ax_dp = construct_nonlinear_model(x_dp,sigma_g,alpha,0,gamma=gamma,clip=clip)
    ggmrf_sum = 0
    for i in range(rows):
      for j in range(cols):
        for di in range(-1,2):
          for dj in range(-1,2):
            ggmrf_sum += g[di+1,dj+1] * abs(x_dp[i,j]-x_dp[(i+di)%rows,(j+dj)%cols])**p
    # divide by 2 because we counted each clique twice
    ggmrf_sum /= 2.
    cost_prior_dp = ggmrf_sum/(p*sigx**p)
    admm_cost_dp.append(sum(sum((y-Ax_dp)**2))/(2*sigw*sigw) + beta*cost_prior_dp)
    diff_img = x - x_dp
    # update u
    u = u+(x-v)
    u_dp = u_dp+(x_dp-v_dp)
    err_img = x-z
    err_img_dp = x_dp-z
    residual_x.append(((x-x_old)**2).mean(axis=None)) 
    residual_v.append(((v-v_old)**2).mean(axis=None)) 
   
    # process plots and images for current iteration 
    imsave(os.path.join(output_dir,'grad_output_itr_'+str(itr)+'.png'),np.clip(x,0,1)) 
    imsave(os.path.join(output_dir,'dp_output_itr_'+str(itr)+'.png'),np.clip(x_dp,0,1)) 

    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_img, cmap='coolwarm',vmin=-err_img.max(),vmax=err_img.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'grad_err_itr'+str(itr+1)+'.png'))
    plt.close() 
   
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_img_dp, cmap='coolwarm',vmin=-err_img.max(),vmax=err_img.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'dp_err_itr'+str(itr+1)+'.png'))
    plt.close() 
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(diff_img, cmap='coolwarm',vmin=-diff_img.max(),vmax=diff_img.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'diff_x_itr'+str(itr+1)+'.png'))
    plt.close() 

  ################ end plug and play ADMM recursive update
  # variance of u
  sig_u_dp = sqrt((u_dp**2).mean(axis=None))
  print("sigma of u_dp: ",sig_u_dp)
  
  #Data post-processing: results visualization 
  plt.figure()
  plt.hist(u_dp.flatten(),bins=50,range=(-u_dp.max(),u_dp.max()),label="$\sigma_u=$"+str(sig_u_dp))
  plt.xlabel("pixel value")
  plt.ylabel("number of pixels")
  plt.legend(loc='upper right')
  plt.savefig(os.path.join(output_dir,'u_histogram.png'))
  plt.close()
  plt.figure()
  plt.semilogy(list(range(residual_x.__len__())), residual_x, label="$log\{\dfrac{1}{N}||x^{n+1}-x^n||^2\}$")
  plt.semilogy(list(range(residual_v.__len__())), residual_v, label="$log\{\dfrac{1}{N}||v^{n+1}-v^n||^2\}$")
  plt.legend(loc='upper right')
  plt.xlabel('itr')
  plt.ylabel('residual')
  plt.savefig(os.path.join(output_dir,'residual.png')) 
  plt.close()
  figout = 'pnp_output_nonlinear.png'
  imsave(os.path.join(output_dir,figout),np.clip(x,0,1))
  plt.figure()
  plt.semilogy(list(range(admm_cost.__len__())), admm_cost,label="pnp with gradient calculation")
  plt.semilogy(list(range(admm_cost_dp.__len__())), admm_cost_dp,label="pnp with deep proximal map")
  plt.legend(loc='upper right')
  plt.xlabel('itr')
  plt.ylabel('admm cost')
  plt.savefig(os.path.join(output_dir,'admm_cost.png')) 
  plt.close()
  return


