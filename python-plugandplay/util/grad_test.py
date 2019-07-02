import numpy as np
import sys,os
sys.path.append(os.path.join(os.getcwd(), "../denoisers/DnCNN"))
from skimage.io import imread, imsave
from construct_forward_model import construct_nonlinear_model
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
  model_name = "model_nonlinear_noiseless_clip"
else:
  model_name = "model_nonlinear_noiseless_noclip"
print("deep pmap model: ",model_name)
json_file = open(os.path.join(model_dir, model_name+'.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join(model_dir,model_name+'.h5'))
z = np.array(imread('../data/'+fig_in+'.png'), dtype=np.float32) / 255.
[rows,cols] = np.shape(z)
x = copy.deepcopy(z)
#x = np.random.rand(rows,cols)
#x = np.zeros((rows,cols))
Ax = construct_nonlinear_model(x, sigma_g, alpha, 0, gamma=gamma, clip=clip)
y = construct_nonlinear_model(z, sigma_g, alpha, 0, gamma=gamma, clip=clip)
#y = copy.deepcopy(z)
gradf_tf = grad_nonlinear_tf(x,y,sigma_g,alpha,sigw,gamma,clip=clip)
sig_gradf_tf = -sig*sig*gradf_tf

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(Ax, cmap='coolwarm',vmin=0,vmax=1)
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join('Ax.png'))

gradf = grad_nonlinear(x,y,sigma_g,alpha,sigw,gamma,clip=clip)
sig_gradf = -sig*sig*gradf

H = pseudo_prox_map_nonlinear(np.subtract(y,Ax),x,model)
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(H, cmap='coolwarm',vmin=-sig_gradf.max(), vmax=sig_gradf.max())
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join('H.png'))

# plot -sig*sig*gradf
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(sig_gradf, cmap='coolwarm',vmin=-sig_gradf.max(), vmax=sig_gradf.max())
#im = ax.imshow(gradf, cmap='coolwarm')
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join('gradf.png'))
# plot grad
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
im = ax.imshow(sig_gradf_tf, cmap='coolwarm',vmin=-sig_gradf.max(), vmax=sig_gradf.max())
#im = ax.imshow(gradf_tf, cmap='coolwarm')
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.savefig(os.path.join('gradf_tf.png'))
