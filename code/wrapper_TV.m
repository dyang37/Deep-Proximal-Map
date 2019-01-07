function out = wrapper_TV(in,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = wrapper_TV(in,sigma)
% performs total variation denoising
% 
% Require deconvtv package
%
% Download:
% http://www.mathworks.com/matlabcentral/fileexchange/43600
%
% Xiran Wang and Stanley Chan
% Copyright 2016
% Purdue University, West Lafayette, In, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tmp  = deconvtv(in,1,1/sigma^2);
out  = tmp.f;

end