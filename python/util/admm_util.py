from math import floor
import matplotlib.pyplot as plt
import numpy as np
import copy
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
  '''
  plt.figure()
  plt.subplot(121)
  plt.imshow(image,interpolation='nearest',cmap='gray')
  plt.title('original image')
  plt.subplot(122)
  plt.imshow(x,interpolation='nearest',cmap='gray')
  plt.title('sheppard interpolated image')
  plt.show()
  '''
  return x
