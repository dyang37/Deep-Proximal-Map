import numpy as np

def rowproj(v,y,siglambd, sigy):
  e = y - v.mean(axis=1)
  (rows,cols) = np.shape(v)
  Ate = np.zeros((rows,cols))
  for j in range(cols):
    Ate[:,j] = e
  x = v + Ate * (siglambd*siglambd)/(siglambd*siglambd/cols + sigy*sigy)
  return x

def colproj(v,y,siglambd, sigy):
  e = y - v.mean(axis=0)
  (rows,cols) = np.shape(v)
  Ate = np.zeros((rows,cols))
  for i in range(cols):
    Ate[i,:] = e
  x = v + Ate * (siglambd*siglambd)/(siglambd*siglambd/rows + sigy*sigy)
  return x

