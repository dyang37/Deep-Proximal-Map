import numpy as np

def icd_rowproj(x,v,y,sig=1.):
  # row projection
  e = y-x.sum(axis=1)
  (r,w) = np.shape(x)
  for i in range(r):
    for j in range(w):
      alpha = (e[i] - (x[i,j]-v[i,j])/(sig*sig))/(1.+1/(sig*sig))
      #print("(",i,",",j,"), alpha=",alpha)
      x[i,j] += alpha
      e[i] -= alpha
  return x

def icd_colproj(x,v,y,sig=1.):
  # row projection
  e = y-x.sum(axis=0)
  (r,w) = np.shape(x)
  for i in range(r):
    for j in range(w):
      alpha = (e[j] - (x[i,j]-v[i,j])/(sig*sig))/(1.+1/(sig*sig))
      #print("(",i,",",j,"), alpha=",alpha)
      x[i,j] += alpha
      e[j] -= alpha
  return x
