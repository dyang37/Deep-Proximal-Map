# Python Plug-and-play Package for Image Processing

Diyu Yang and Prof. Charles A. Bouman
Purdue University, School of Electrical and Computer Engineering
## Overview
This python package implements image inpainting and super-resolution with different choices of prior models. 

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




