import numpy as np
import sys,os,argparse
sys.path.append(os.path.join(os.getcwd(), "../util"))
from ml_estim_mnist import ml_estimate_mnist

parser = argparse.ArgumentParser()
parser.add_argument('--d', default=0, type=int, help='digit to generate')
args = parser.parse_args()

################### hyperparameters
sigw = 0.
sig = "var"
################### Data Proe-processing
y = np.zeros((10,))
y[args.d] = 1.
y += np.random.normal(0,sigw,y.shape)
################## Forward model construction
ml_estimate_mnist(y,sig,sigw,args.d)
