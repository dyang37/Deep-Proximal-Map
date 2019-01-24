import numpy as np
import copy
'''
def generateAmatrix(image, indicator):
  return np.array(np.where(image != indicator))
'''

def DataTermOptSR(y, map_image, icd_niter, u, v, lambd, mu, R):
  new_image = copy.copy(map_image)
  for k in range(icd_niter):
    new_image = SR_ADMMUpdate(y, new_image, u, v, lambd, mu, R)
  return new_image


def SR_ADMMUpdate(y, map_image, u, v, lambd, mu, R):
  x_tilde = v - u
  [m, n] = np.shape(map_image)
  ret_mapimg = np.zeros((m,n))
  for row in range(m):
    for col in range(n):
      y_row = row//R
      y_col = col//R
      update = x_tilde[row,col] + lambd*(y[y_row,y_col]-mu[y_row,y_col])
      ret_mapimg[row,col] = max(0,update)
  return ret_mapimg




