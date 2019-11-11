import os,sys
import numpy as np
import json

outdir_name = "MACE5"
err_samples = 0
tol = 0.2
for idx in range(-1000,0):
  exp_dir = os.path.join(os.getcwd(),'../results/mace_mnist_cnn/'+outdir_name)
  result_dir = os.path.join(exp_dir,"idx"+str(idx))
  with open(os.path.join(result_dir,'statistics.txt')) as json_file:
    [_,mse_x,mse_ynorm2] = json.load(json_file)
    if mse_ynorm2 > tol:
      print("idx = ",idx,": error case, mse of norm2 y = ",mse_ynorm2)
      err_samples += 1

print("total error samples: ",err_samples)
