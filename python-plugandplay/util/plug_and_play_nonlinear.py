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
from forward_model import camera_model
from grad import grad_nonlinear_tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from skimage.restoration import denoise_tv_chambolle as denoiser_tv
from math import sqrt

def plug_and_play_nonlinear(p,z,y,sigma_g,alpha,beta,sigw,sigw_train,sig,gamma,clip):
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
  output_dir = os.path.join(output_dir, "sigwTrain_"+str(sigw))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  # save input y 
  imsave(os.path.join(output_dir,'y_input.png'), np.clip(y,0,1))
  
  # parse denoiser model dir
  denoiser_dir="/home/yang1467/deepProxMap/my_plug_and_play/python-plugandplay/denoisers/DnCNN/denoiserModels"
  denoiser_name = "dncnn_sigma0.1"
  json_file = open(os.path.join(denoiser_dir,denoiser_name+'.json'), 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  denoiser_model = model_from_json(loaded_model_json)
  denoiser_model.load_weights(os.path.join(denoiser_dir,denoiser_name+'.h5'))
  
  # parse deep proximal map model dir
  pmap_dir=os.path.join(os.getcwd(),'../cnn/nonlinear_model')
  pmap_model_name = "model_nonlinear_"
  if clip:
    pmap_model_name += "clip_sigw_"
  else:
    pmap_model_name += "noclip_sigw_"
  pmap_model_name += str(sigw_train)
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
  diff_mse = []
  sig_alpha = sig/sigw_train
  # iterative reconstruction
  init_itr = 10
  max_itr = 40
  for itr in range(init_itr):
    xtilde = np.subtract(v,u)
    Axtilde = camera_model(xtilde,sigma_g,alpha,0,gamma=gamma, clip=clip)
    H = pseudo_prox_map_nonlinear(np.subtract(y,Axtilde),xtilde,pmap_model)
    x = np.add(xtilde, H)
    vtilde = np.add(x,u)
    sig_n = sqrt(beta)*sig
    if p==1:
      v = denoiser_tv(vtilde, sig_n)
    elif p==2:
      v = gmrf_denoiser(v,vtilde,sig_n,sigx)
    else:
      v = cnn_denoiser(vtilde, denoiser_model)
    u = u+(x-v)
  
  v_dp = copy.deepcopy(v)
  v_cp = copy.deepcopy(v)
  x_dp = copy.deepcopy(x)
  x_cp = copy.deepcopy(x)
  u_dp = copy.deepcopy(u)
  u_cp = copy.deepcopy(u)
  for itr in range(init_itr,max_itr):
    print("iteration ",itr)
    # forward step
    xtilde_dp = np.subtract(v_dp,u_dp)
    xtilde_cp = np.subtract(v_cp,u_cp)
    imsave(os.path.join(output_dir,'xtilde_itr_'+str(itr)+'.png'),np.clip(xtilde_dp,0,1)) 
    
    Axtilde_dp = camera_model(xtilde_dp,sigma_g,alpha,0,gamma=gamma, clip=clip)
    Axtilde_cp = camera_model(xtilde_cp,sigma_g,alpha,0,gamma=gamma, clip=clip)
    H = pseudo_prox_map_nonlinear(np.subtract(y,Axtilde_dp),xtilde_dp,pmap_model)
    grad_f_tf = grad_nonlinear_tf(xtilde_cp,y,sigma_g,alpha,1,gamma,clip=clip)
    sig_gradf_tf = -sig_alpha*sig_alpha*grad_f_tf
    x_dp = np.add(xtilde_dp, H)
    x_dp_check = np.add(xtilde_dp, -sig_alpha*sig_alpha*grad_nonlinear_tf(xtilde_dp,y,sigma_g,alpha,1,gamma,clip=clip))
    x_cp = np.add(xtilde_cp, sig_gradf_tf)
    vtilde_dp = np.add(x_dp,u_dp)
    vtilde_cp = np.add(x_cp,u_cp)
    # denoising step
    sig_n = sqrt(beta)*sig
    if p==1:
      v_dp = denoiser_tv(vtilde_dp, sig_n)
      v_cp = denoiser_tv(vtilde_cp, sig_n)
    elif p==2:
      v_dp = gmrf_denoiser(v_dp,vtilde_dp,sig_n,sigx)
      v_cp = gmrf_denoiser(v_cp,vtilde_cp,sig_n,sigx)
    else:
      v_dp = cnn_denoiser(vtilde_dp, denoiser_model)
      v_cp = cnn_denoiser(vtilde_cp, denoiser_model)


    diff_img = x_dp - x_cp
    diff_mse.append(sqrt((diff_img**2).mean(axis=None)))
    # update u
    u_dp = u_dp+(x_dp-v_dp)
    u_cp = u_cp+(x_cp-v_cp)
    err_img_dp = x_dp-z
    err_img_cp = x_cp-z
   
    # process plots and images for current iteration 
    imsave(os.path.join(output_dir,'deep_prox_map_itr_'+str(itr)+'.png'),np.clip(x_dp,0,1)) 
    imsave(os.path.join(output_dir,'check_deep_prox_map_itr_'+str(itr)+'.png'),np.clip(x_dp_check,0,1)) 
    imsave(os.path.join(output_dir,'conventional_prox_map_itr_'+str(itr)+'.png'),np.clip(x_cp,0,1)) 

    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_img_dp, cmap='coolwarm',vmin=-err_img_dp.max(),vmax=err_img_dp.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'dp_err_itr'+str(itr)+'.png'))
    plt.close() 
    
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(err_img_cp, cmap='coolwarm',vmin=-err_img_dp.max(),vmax=err_img_dp.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'cp_err_itr'+str(itr)+'.png'))
    plt.close() 
  
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    im = ax.imshow(diff_img, cmap='coolwarm',vmin=-diff_img.max(),vmax=diff_img.max())
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.savefig(os.path.join(output_dir,'diff_x_itr'+str(itr)+'.png'))
    plt.close() 

  #Data post-processing: results visualization 
  figout = 'pnp_output_nonlinear.png'
  imsave(os.path.join(output_dir,figout),np.clip(x,0,1))
  plt.figure()
  plt.semilogy(list(range(init_itr,max_itr)), diff_mse)
  plt.xlabel('itr')
  plt.ylabel('$\sqrt{\dfrac{1}{N}||x_dp-x_cp||^2}$')
  plt.savefig(os.path.join(output_dir,'diff_mse.png')) 
  plt.close()

  
  return


