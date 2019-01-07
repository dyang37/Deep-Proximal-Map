function out = wrapper_cnn(in,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = wrapper_cnn(in,sigma)
% performs deep neural network denoising
% 
% Diyu Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net = denoisingNetwork('DnCNN');
out = denoiseImage(in, net);

end