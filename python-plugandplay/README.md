# Python Plug-and-play Package for Image Processing

Diyu Yang and Prof. Charles A. Bouman
Purdue University, School of Electrical and Computer Engineering
## Overview
This python package implements image inpainting and super-resolution with different choices of prior models. For more details please refer to pnp_super_resol.pptx. 

## Prerequisite
Python 3.6

Keras 2.2.4

Tensorflow 1.12.0 (GPU support is not necessary)

scikit-image

matplotlib

## Running the demos:
1. Move to Plugandplay-python directory

2. run `python demo_inpaint.py <prior>`. Where `<choice>` is the prior model option. See [Prior model choices](#Prior-model-choices) for further details.
Replace `demo_inpaint.py` with `demo_sr.py` for super-resolution demo.

3. Output image will be saved in Plugandplay-python directory.

## Prior model choices 
0: Convolutional Neural Network ([Source Code](https://github.com/cszn/DnCNN), [Paper](https://arxiv.org/pdf/1608.03981.pdf))

1: Total Variation

2: Non-local Mean ([Paper](https://ieeexplore.ieee.org/document/1467423))

## Parameter Tuning

### Default Paramters:

Maximum iteration: 200 for SR/100 for inpainting

gamma: 0.99 for SR/1 for inpainting    update factor for rho

lambda: 0.01    Regularizer for image denoiser

rho: 1    Adaptive update rule for the residual

nl_mean denoiser: patch_distance = 11, patch_size = 7

## Current issues to be addressed:
For the super resolution problem, the aliasing effects increases rapidly for the case of large subsampling factors (8x and more). Initial guess is that the cutoff frequency for our anti-aliasing filter (currently a gaussian filter with variance 1) is above \pi/8, therefore causing aliasing effects in high frequency components. 
