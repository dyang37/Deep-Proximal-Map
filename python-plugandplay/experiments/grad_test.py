import numpy as np
import random
import sys,os
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
sys.path.append(os.path.join(os.getcwd(), "../util"))
from math import sqrt
from skimage.io import imread, imsave
from forward_model import camera_model
from grad import grad_nonlinear, grad_nonlinear_tf
from dncnn import pseudo_prox_map_nonlinear
from keras.models import model_from_json
import matplotlib.pyplot as plt
import copy

clip = False
sigw = 0.05
sig = 0.05
sigma_g = 10
alpha = 0.5
gamma = 2.
fig_in = 'test_gray'

model_dir=os.path.join(os.getcwd(),'../cnn')
if clip:
  model_name = "model_nonlinear_noisy_clip"
else:
  model_name = "model_nonlinear_noisy_noclip"
print("deep pmap model: ",model_name)
json_file = open(os.path.join(model_dir, model_name+'.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join(model_dir,model_name+'.h5'))
z = np.array(imread('../'+fig_in+'.png'), dtype=np.float32) / 255.
[rows,cols] = np.shape(z)
#x = copy.deepcopy(z)
x = np.ones((rows,cols))/2.
#x = np.random.rand(rows,cols)
Ax = camera_model(x, sigma_g, alpha, 0, gamma=gamma, clip=clip)
y = camera_model(z, sigma_g, alpha, 0, gamma=gamma, clip=clip)
gradf_tf = grad_nonlinear_tf(x,y,sigma_g,alpha,sigw,gamma,clip=clip)
gradf = grad_nonlinear(x,y,sigma_g,alpha,sigw,gamma,clip=clip)
        

sig_gradf_tf = -sig*sig*gradf_tf
sig_gradf = -sig*sig*gradf
# finite difference method for gradient calculation

H = pseudo_prox_map_nonlinear(np.subtract(y,Ax),x,model)

print("mse=",sqrt(((H-sig_gradf_tf)**2).mean()))
# finite difference method to verify gradient calculation
epsil = 0.005
for i in random.sample(list(range(512)), 3):
  for j in random.sample(list(range(512)), 3):
    x_plus = copy.deepcopy(x)
    x_minus = copy.deepcopy(x)
    x_plus[i,j] += epsil
    x_minus[i,j] -= epsil
    Ax_plus = camera_model(x_plus, sigma_g, alpha, 0, gamma=gamma, clip=clip)
    Ax_minus = camera_model(x_minus, sigma_g, alpha, 0, gamma=gamma, clip=clip)
    fx_plus = np.sum((y-Ax_plus)**2)/(2.*sigw*sigw)
    fx_minus = np.sum((y-Ax_minus)**2)/(2.*sigw*sigw)
    grad_s = (fx_plus-fx_minus)/(2.*epsil)
    print("pixel (",i,j,")")
    print("tf grad:",sig_gradf_tf[i,j],"; FD grad:", -sig*sig*grad_s,"; H:", H[i,j])


fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(H, cmap='coolwarm',vmin=-H.max(), vmax=H.max())
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join('H.png'))

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(sig_gradf, cmap='coolwarm',vmin=-H.max(), vmax=H.max())
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join('gradf.png'))

# plot grad
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(sig_gradf_tf, cmap='coolwarm',vmin=-H.max(), vmax=H.max())
#im = ax.imshow(gradf_tf, cmap='coolwarm')
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join('gradf_tf.png'))
