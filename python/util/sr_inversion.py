import numpy as np
import copy
def sr_inversion(mu,y,xtilde,R,niter, lambd):
  [rows_hr, cols_hr] = np.shape(xtilde)
  x = np.zeros((rows_hr,cols_hr))
  for itr in range(niter):
    for row in range(rows_hr):
      for col in range(cols_hr):
        x[row,col] = icd_update(xtilde,y,mu,row,col,R,lambd)
  return x


def icd_update(xtilde,y,mu,row_hr,col_hr,R,lambd):
  row_lr = row_hr//R
  col_lr = col_hr//R
  ys = y[row_lr,col_lr]
  xtilde_s = xtilde[row_hr,col_hr]
  mu_s = mu[row_lr,col_lr]
  xs = xtilde_s + lambd*ys - mu_s
  return np.clip(xs,0,1)

'''
def fetch_neighs(x,R,row,col):
  [h,w] = np.shape(x)
  row_i = row%R
  col_j = col%R
  neigh_list = []
  for i in range(row//R*R,(row//R+1)*R):
    for j in range(col//R*R,(col//R+1)*R):
      if (i==row_i) and (j==col_j):
        continue
      # applying circular boundary condition
      neigh_list.append(x[i%h,j%h])
  return neigh_list
'''

