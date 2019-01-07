function out = wrapper_RF(in,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = wrapper_RF(in,sigma)
% performs recursive filter denoising
% 
% Require RF package
% 
% Available in ./RF/
%
% Xiran Wang and Stanley Chan
% Copyright 2016
% Purdue University, West Lafayette, In, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out = RF(in, 3, sigma, sigma, 3);

end