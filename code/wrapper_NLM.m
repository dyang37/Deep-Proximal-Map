function out = wrapper_NLM(in,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = wrapper_NLM(in,sigma)
% performs non-local means denoising
% 
% Require NLM package
%
% Download:
% http://www.ipol.im/pub/art/2011/bcm_nlm/
%
% Xiran Wang and Stanley Chan
% Copyright 2016
% Purdue University, West Lafayette, In, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Options.filterstrength=sigma;
out = NLMF(in,Options);
end