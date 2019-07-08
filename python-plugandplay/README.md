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

## Code structure
### experiments/
This directory contains all the experiment code for both linear and nonlinear forward model case.

For experiments with linear forward model, an image filter-downsampling model is used.

For experiments with nonlinear forward model, a blurry camera model is used.

* #### ml_exp_linear.py: 

  Wrapper for linear case Pseudo ML estimation.
  output image files are in results/ml_output_linear/ directory

* #### ml_exp_nonlinear.py: 

  Wrapper for nonlinear case Pseudo ML estimation.
  output image files are in results/ml_output_nonlinear/ directory
  
* #### pnp_linear.py: Wrapper for plug and play reconstruction experiment of linear forward model.
* #### pnp_linear.py: Wrapper for plug and play reconstruction experiment of nonlinear forward model.


