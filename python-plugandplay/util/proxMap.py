import numpy as np
from forward_model import blockAvgModel
def PM_rowproj(v,y,siglambd, sigy):
  e = y - v.mean(axis=1)
  (rows,cols) = np.shape(v)
  Ate = np.zeros((rows,cols))
  for j in range(cols):
    Ate[:,j] = e
  x = v + Ate * (siglambd*siglambd)/(siglambd*siglambd/cols + sigy*sigy)
  return x

def PM_colproj(v,y,siglambd, sigy):
  e = y - v.mean(axis=0)
  (rows,cols) = np.shape(v)
  Ate = np.zeros((rows,cols))
  for i in range(cols):
    Ate[i,:] = e
  x = v + Ate * (siglambd*siglambd)/(siglambd*siglambd/rows + sigy*sigy)
  return x

def PM_blockavg(v,y,siglambd,sigy,L=2):
  Av = blockAvgModel(v,L=L)
  e = y - Av
  (rows,cols) = np.shape(v)
  Ate = np.zeros((rows,cols))
  for i in range(rows):
    for j in range(cols):
      Ate[i][j] = e[i//L][j//L]/(L*L)
  x = v + Ate * (siglambd*siglambd)/(siglambd*siglambd/(L*L) + sigy*sigy)
  return x

