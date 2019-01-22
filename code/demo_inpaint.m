%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo of Plug-and-Play ADMM for image inpainting
%
% S. H. Chan, X. Wang, and O. A. Elgendy
% "Plug-and-Play ADMM for image restoration: Fixed point convergence
% and applications", IEEE Transactions on Computational Imaging, 2016.
% 
% ArXiv: https://arxiv.org/abs/1605.01710
% 
% Xiran Wang and Stanley Chan
% Copyright 2016
% Purdue University, West Lafayette, In, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

addpath(genpath('./utilities/'));

%add path to denoisers
addpath(genpath('/Users/dyangsteven/Plug_n_play/denoisers/BM3D/'));
addpath(genpath('/Users/dyangsteven/Plug_n_play/denoisers/TV/'));
addpath(genpath('/Users/dyangsteven/Plug_n_play/denoisers/NLM/'));
addpath(genpath('/Users/dyangsteven/Plug_n_play/denoisers/RF/'));
%addpath(genpath('/Users/dyangsteven/Plug_n_play/CNN_code/'));
%read test image
z = im2double(imread('./data/Art512.png'));


%reset random number generator 
rng(0)

%initialize a mask for sampling
mask = rand(size(z))>=0.8;

%set noies level
noise_level = 10/255;
z_mask = z.*mask;
%calcualte the observed image
y = z_mask + noise_level*randn(size(z));
%
%parameters
method = 'cnn';
switch method
    case 'RF'
        lambda = 0.0003;
    case 'NLM'
        lambda = 0.002;
    case 'BM3D'
        lambda = 0.001;
    case 'TV'
        lambda = 0.01;
    case 'cnn'
        lambda = 0.01;
end

%optional parameters
opts.rho     = 1;
opts.gamma   = 1;
opts.max_itr = 5;
opts.print   = true;

%main routine
tic
out = PlugPlayADMM_inpaint(y,mask,lambda,method,opts);
toc

%display
PSNR_output = psnr(out,z);
fprintf('\nPSNR = %3.2f dB \n', PSNR_output);

figure;
subplot(131);
imshow(z);
title('ground truth');
subplot(132);
imshow(y);
title('noisy input');
subplot(133);
imshow(out);
tt = sprintf('PSNR = %3.2f dB', PSNR_output);
title(tt);

