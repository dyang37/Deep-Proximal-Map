import numpy as np
import copy
from math import floor

def generateAmatrix(image, indicator):
  return np.array(np.where(image != indicator))

def DataTermOptInPaint(map_image, sampled, n_iter, u, v, lambd, Amtx):
  Hx = map_image
  map_x = copy.copy(map_image)
  Mask =  np.zeros(np.shape(sampled))
  listLen = np.shape(Amtx)[0]
  e = np.zeros(np.shape(sampled))
  for i in range(listLen):
    row = Amtx[i,0]
    col = Amtx[i,1]
    e[row,col] = sampled[row,col] - Hx[row,col]
    Mask[row,col] = 1
  for k in range(n_iter):
    [map_x, e] = InPaintADMMUpdate(map_x, Mask, e, 0, u, v, lambd)
  return map_x


def InPaintADMMUpdate(map_image,Mask, e, var_w, u, v, lambd):
  TempK = v - u
  [m,n] = np.shape(map_image)
  for row in range(m):
    for col in range(n):
      v_temp = copy.copy(map_image[row,col])
      if(Mask[row,col] == 1):
        update = (var_w * lambd * TempK[row,col] + (e[row,col] + map_image[row,col])) / (1 + var_w * lambd)
      else:
        update = TempK[row, col]

      if(update > 0):
        map_image[row,col] = update;
      else:
        map_image[row,col] = 0
      e[row,col] = e[row,col] - (map_image[row,col]-v_temp)
  return [map_image, e] 

def shepard_init(image, measurement_mask, window):
  p = 2
  wing = floor(window/2.)
  [h,w]=np.shape(image)
  x = copy.copy(image)
  for i in range(0,h):
    i_lower_limit = -min(wing, i)
    i_upper_limit = min(wing, h-i)
    for j in range(0,w):
      if (measurement_mask[i,j] == 0):
        IPD = np.zeros(window*window)
        pixel = np.zeros(window*window)
        j_lower_limit = -min(wing,j)
        j_upper_limit = min(wing,w-j)
        count = 0
        sum_IPD = 0
        interpolated_value = 0
        for neigh_i in range(i+i_lower_limit, i+i_upper_limit):
          for neigh_j in range(j+j_lower_limit, j+j_upper_limit):
            if (measurement_mask[neigh_i,neigh_j] == 1):
              count = count + 1
              IPD[count] = float(1./((neigh_i - i)**p + (neigh_j - j) ** p))
              sum_IPD = sum_IPD + IPD[count]
              pixel[count] = copy.copy(image[neigh_i,neigh_j])

        for c in range(1,count+1):
          weight = IPD[c] / sum_IPD
          interpolated_value = float(interpolated_value + weight*pixel[c])
        x[i,j] = interpolated_value
  return x

