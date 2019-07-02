import os,sys,glob
from skimage.io import imread
sys.path.append(os.path.join(os.getcwd(), "../util"))
import numpy as np
import pickle
import matplotlib.pyplot as plt

# allow GPU growth

def to_gray(x_in):
  [rows,cols] = np.shape(x_in)[0:2]
  x = np.zeros((rows, cols))
  for i in range(rows):
    for j in range(cols):
      r = x_in[i,j,0]
      g = x_in[i,j,1]
      b = x_in[i,j,2]
      x[i,j]=0.2989 * r + 0.5870 * g + 0.1140 * b
  return x

v = []
n_samples = 0

for filename in glob.glob('/root/datasets/DIV2K_*_HR/*.png'):
  #print(filename)
  n_samples += 1
  v_in = np.array(imread(filename), dtype=np.float32) / 255.0
  v_img = to_gray(v_in)
  [rows_hr,cols_hr] = v_img.shape
  for i in range(0,rows_hr-512,512):
    for j in range(0,cols_hr-512,512):
      v_crop = v_img[i:i+512,j:j+512]
      v.append(v_crop)

print(np.shape(v))
dataset = {'v':v}
dict_name = '/root/datasets/raw_input.dat'
fd = open(dict_name,'wb')
pickle.dump(dataset, fd)
fd.close()
print('Done')
