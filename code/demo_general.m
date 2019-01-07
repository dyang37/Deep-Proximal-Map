%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo of Plug-and-Play ADMM for general image restoration
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
z = im2double(imread('./data/House256.png'));

%construct A matrix, deblurring as an example
dim = size(z);
h = fspecial('gaussian',[9 9],1);
A = @(z,trans_flag) afun(z,trans_flag,h,dim);

%reset random number generator
rng(0);

%set noies level
noise_level = 10/255;

%calculate observed image
y = A(z(:),'transp') + noise_level*randn(prod(dim),1);
y = proj(y,[0,1]);
y = reshape(y,dim);

%parameters
method = 'RF';
switch method
    case 'RF'
        lambda = 0.0005;
    case 'NLM'
        lambda = 0.005;
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
out = PlugPlayADMM_general(y,A,lambda,method,opts);

%display
PSNR_output = psnr(out,z);
fprintf('\nPSNR = %3.2f dB \n', PSNR_output);

figure;
subplot(121);
imshow(y);
title('Input');

subplot(122);
imshow(out);
tt = sprintf('PSNR = %3.2f dB', PSNR_output);
title(tt);
