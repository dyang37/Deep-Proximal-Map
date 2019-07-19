import numpy as np
g = np.array([[1./12., 1./6., 1./12.],[1./6.,0,1./6.],[1./12.,1./6.,1./12.]])

def gmrf_denoiser(x,xtilde,sig_n, sig_x):
  [rows, cols] = np.shape(x)
  for i in range(rows):
    for j in range(cols):
      xs = x[i,j]
      neigh_sum = 0
      for di in range(-1,2):
        for dj in range(-1,2):
          neigh_sum += g[di+1,dj+1] * x[(i-di)%rows,(j-dj)%cols]
      x[i,j] = np.clip((xtilde[i,j]+neigh_sum*sig_n*sig_n/(sig_x*sig_x))/(1.+sig_n*sig_n/(sig_x*sig_x)),0,None)
  return x
