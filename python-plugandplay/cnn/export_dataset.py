import os,sys
sys.path.append(os.path.join(os.getcwd(), "../util"))
import numpy as np
import pickle
# allow GPU growth
from construct_forward_model import construct_nonlinear_model


dict_name = '/root/datasets/raw_input.dat'
dataset = pickle.load(open(dict_name,"rb"))
v = np.array(dataset['v'])
print(np.shape(v))
[n_samples,rows,cols] = np.shape(v)
print('total number of samples: ',n_samples)

# parameters
clip = False
sig = 0.05
sigw = 0.
sigma_g = 10
alpha = 0.5
gamma = 2.
forward_name = 'nonlinear'

epsil = []
y_Av = []
for v_img in v:
  epsil_k = np.random.normal(0,sig,v_img.shape)
  x_img = np.add(v_img, epsil_k)
  y = construct_nonlinear_model(x_img, sigma_g, alpha, sigw, gamma=gamma,clip=clip)
  Av = construct_nonlinear_model(v_img, sigma_g, alpha, 0, gamma=gamma,clip=clip)
  y_Av.append(y-Av)
  epsil.append(epsil_k) 
# Random Shuffle and training/test set selection
epsil  = np.array(epsil)
y_Av = np.array(y_Av)

dict_num = 0
for i in [0,2000,4000]:
  new_dataset = {"v":v[i:min(n_samples,i+2000)], "y_Av":y_Av[i:min(n_samples,i+2000)], "epsil":epsil[i:min(n_samples,i+2000)]}
  new_dict = '/root/datasets/noiseless_nonlinear_dict'+str(dict_num)+'.dat'
  fd = open(new_dict,'wb')
  pickle.dump(new_dataset, fd)
  fd.close()
  dict_num+=1
print("processed dataset exported")

