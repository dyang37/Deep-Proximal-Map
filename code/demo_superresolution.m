%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo of Plug-and-Play ADMM for image super-resolution
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
addpath(genpath('./denoisers/BM3D/'));
addpath(genpath('./denoisers/TV/'));
addpath(genpath('./denoisers/NLM/'));
addpath(genpath('./denoisers/RF/'));

%read test image
z = im2double(imread('./data/Couple512.png'));

%blur kernel and downsampling factor
h = fspecial('gaussian',[9 9],1);
K = 2;
noise_level = 10/255;
rng(0)
%calculate the observed image
y = imfilter(z,h,'circular');
y = downsample2(y,K);
y = y + noise_level*randn(size(y));

%parameters
method = 'BM3D';
switch method
    case 'RF'
        lambda = 0.0002;
    case 'NLM'
        lambda = 0.001;
    case 'BM3D'
        lambda = 0.001;
    case 'TV'
        lambda = 0.01;
end

%optional parameters
opts.rho     = 1;
opts.gamma   = 1;
opts.max_itr = 20;
opts.print   = true;

%main routine
tic
out = PlugPlayADMM_super(y,h,K,lambda,method,opts);
toc
%display
PSNR_output = psnr(out,z);
fprintf('\nPSNR = %3.2f dB \n', PSNR_output);

figure;
subplot(121);
imshow(imresize(y,K,'nearest'));
title('Input');

subplot(122);
imshow(out);
tt = sprintf('PSNR = %3.2f dB', PSNR_output);
title(tt);
