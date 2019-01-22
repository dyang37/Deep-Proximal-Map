import numpy as np
import copy

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



