# Deep Proximal Map For Inverse Image Problems

Diyu Yang and Prof. Charles A. Bouman

Purdue University, School of Electrical and Computer Engineering
## Overview
This python package implements deep proximal map model and related experiments (ML estimation, Plug and play reconstruction) for various inverse imaging problems.

Note that this repo does NOT include original training data for deep proximal map. If you would like to retrain the model please download the training dataset [HERE](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

## Prerequisite
Python 3.6

Keras 2.2.4

Tensorflow 1.12.0 (GPU support is only necessary if you want to retrain the deep proximal map models)

scikit-image

matplotlib

Cython

## experiments/
This directory contains wrappers for all image reconstruction experiments for both linear and nonlinear forward model case.

For experiments with linear forward model, an image filter-downsampling model is used.

For experiments with nonlinear forward model, a blurry camera model is used.

* #### ml_exp_linear.py: 

  Wrapper for linear case Pseudo ML estimation.
  
  output image files are in results/ml_output_linear/ directory

* #### ml_exp_nonlinear.py: 

  Wrapper for nonlinear case Pseudo ML estimation.
  
  output image files are in results/ml_output_nonlinear/ directory
  
* #### pnp_linear.py: 

  Wrapper for plug and play reconstruction experiment of linear forward model.
  
  Input Arguments:
  
    -f: Forward model optimization method. 0: Deep Proximal Map. 1: ICD. Default choice is 0.
    
    A quick demo: `python pnp_linear.py -f 0`
    
    Output image files are in results/pnp_output/linear/ directory
  
* #### pnp_nonlinear.py: 

  Wrapper for plug and play reconstruction experiment of nonlinear forward model.

  Output image files are in results/pnp_output/nonlinear/ directory
  
* #### grad_test.py:

  Gradient image experiment for nonlinear forward model. Given an input x and y (defined in the code), generate gradient image of ![gradient](https://latex.codecogs.com/gif.latex?%5Cnabla%20f%28x%29) approximated by deep proximal map and by tensorflow respectively, where ![fx](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cfrac%7B1%7D%7B2%7D%7C%7Cy-A%28x%29%7C%7C%5E2_B).
  
  Output gradient images are in experiment/ directory.
  
  
## cnn/

This directory contains all image pre-processing and model training code. It also contains pre-trained deep proximal map models for both linear and nonlinear forward models. 

#### Code

* train_nonlinear.py: training code for nonlinear forward model.

* raw_data_gen.py: crops the training dataset images into 512x512 image patches, and save it as a pickle file `raw_input.dat`.

#### Pre-trained models
Pretrained deep proximal map models are saved as `<model-name>.h5` (contains model weights) and `<model-name>.json` (contains model structures) files. Below are a list of pre-trained models:

* `model_nonlinear_noiseless_clip`: 
  Model for nonlinear forward model case. Model Parameters are listed below:
  
  AWGN standard-deviation: \sigma_w = 0
  
  Standard-deviation of v with respect to x: \sigma = 0.05
  
  Standard-deviation of gaussian filter in nonlinear forward model: \sigma_g = 10
  
  Gamma-correction: \gamma = 2
  
  Low-pass coefficient for forward model: \alpha = 0.5
  
  Clip in forward model: True. `y=A(clip{x;0})`

* `model_nonlinear_noiseless_noclip`: 
  Same as the above model except that Forward model clipping is depreciated. 
  
* `model_sinc_noisy_simple_hr`:
  Model for linear forward model with a sinc anti-aliasing filter. Parameters:
  
  \sigma = 0.2

  \sigma_w = 0.05
  
  Downsampling factor: K = 4
