# Deep Proximal Map For Inverse Image Problems

Diyu Yang and Prof. Charles A. Bouman

Purdue University, School of Electrical and Computer Engineering
## Overview
This python package implements deep proximal map model and related experiments (ML estimation, Plug and play reconstruction) for various inverse imaging problems.

## Prerequisite
Python 3.6

Keras 2.2.4

Tensorflow 1.12.0 (GPU support is only necessary if you want to retrain the deep proximal map models)

scikit-image

matplotlib

Cython

## Running the demos:
1. Move to Plugandplay-python directory

2. Run `./Cython_setup.sh` to set up cpp wrapper for ICD code

3. run `python demo_sr.py -p <prior> -f <forward>`. 

<prior> is the prior model option. See [Prior model choices](#Prior-model-choices) for further details.


<forward> is the optimization method for forward model proximal map update. See [Proximal Map Update Choices](#Proximal-map-optimization-method-choices) for further details.

4. Output image will be saved in Plugandplay-python directory.

## Prior model choices 
0(default option): Convolutional Neural Network ([Source Code](https://github.com/cszn/DnCNN), [Paper](https://arxiv.org/pdf/1608.03981.pdf))

1: Total Variation

2: Non-local Mean ([Paper](https://ieeexplore.ieee.org/document/1467423))

## Proximal map optimization method choices
Optimization method can be changed by modifying parameter "optim_method" at line 24 in demo_sr.py. Below are the available choices:

0(default option): Approximation by Fourier Decomposition ([Paper](https://ieeexplore.ieee.org/document/1467423))

1: Iterative Coordinate Descent(ICD)

Note that ICD code is written in C++ and called by python with Cython. You need to run `./Cython_setup.sh` when using it the first time.

## Parameter Tuning

### Default Paramters:

Maximum iteration: 40 for SR; 100 for inpainting

gamma: 1    update factor for rho

lambda: 50   Regularizer for image denoiser

rho: 1    Adaptive update rule for the residual

nl_mean denoiser: patch_distance = 11, patch_size = 7

## Current issues to be addressed:
For the super resolution problem, the aliasing effects increases rapidly for the case of large subsampling factors (8x and more). Initial guess is that the cutoff frequency for our anti-aliasing filter (currently a gaussian filter with variance 1) is above \pi/8, therefore causing aliasing effects in high frequency components. 
