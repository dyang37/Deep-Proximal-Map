from icdproj import icd_rowproj, icd_colproj
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
from math import sqrt
sigx = 0.1
sigy = 0.1

def icdproj_wrapper(yr,yc,v,x,sig,ce_itr):
  if ce_itr==0:
    nitr = 10
  else:
    nitr = 1
  for itr in range(nitr):
    x = icd_rowproj(x,v,yr,sig=sig)
    x = icd_colproj(x,v,yc,sig=sig)
  return np.clip(x,0,1)
