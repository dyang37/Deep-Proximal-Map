%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  RF  Domain transform recursive edge-preserving filter.
%
%  F = RF(img, sigma_s, sigma_r, num_iterations, joint_image)
%
%  Parameters:
%    img             Input image to be filtered.
%    sigma_s         Filter spatial standard deviation.
%    sigma_r         Filter range standard deviation.
%    num_iterations  Number of iterations to perform (default: 3).
%    joint_image     Optional image for joint filtering.
%
%  Reference to the original Recursive Filter:
%    Domain Transform for Edge-Aware Image and Video Processing
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 30 (2011), Number 4.
%    Proceedings of SIGGRAPH 2011, Article 69.
%
% 
% Modified for image denoising
% Xiran Wang and Stanley Chan
% Copyright 2016
% Purdue University, West Lafayette, In, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function F = RF(img, sigma_s, sigma_r, noise_sigma, num_iterations,joint_image)

I = double(img);

if ~exist('num_iterations', 'var')
    num_iterations = 3;
end

if ~exist('noise_sigma', 'var')
    noise_sigma = 0;
end

if exist('joint_image', 'var') && ~isempty(joint_image)
    J = double(joint_image);
    
    if (size(I,1) ~= size(J,1)) || (size(I,2) ~= size(J,2))
        error('Input and joint images must have equal width and height.');
    end
else
    J = I;
end

[h w num_joint_channels] = size(J);

%% Compute the domain transform (Equation 11 of our paper).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Estimate horizontal and vertical partial derivatives using finite
% differences.
dIcdx = diff(J, 1, 2);
dIcdy = diff(J, 1, 1);

dIdx = zeros(h,w);
dIdy = zeros(h,w);

% Compute the l1-norm distance of neighbor pixels.
for c = 1:num_joint_channels
    dIdx(:,2:end) = dIdx(:,2:end) + abs( dIcdx(:,:,c) );
    dIdy(2:end,:) = dIdy(2:end,:) + abs( dIcdy(:,:,c) );
end


%XW and SC: include patch smoothing
%choice_one
win = 7;
filter=fspecial('gaussian',[win win],2);
dIdx = imfilter(dIdx, filter, 'replicate');
dIdy = imfilter(dIdy, filter, 'replicate');
%choice_two
% dIdx = FbilateralFilter(dIdx,[],-1,1,3,0.19);
% dIdy = FbilateralFilter(dIdy,[],-1,1,3,0.19);
%--------------------------------------------


%XW and SC: updated derivative by subtracting noise
dHdx = (1 + sigma_s/sigma_r * max(dIdx-noise_sigma,0));
dVdy = (1 + sigma_s/sigma_r * max(dIdy-noise_sigma,0));


%XW and SC: remove cumsum as they are not needed for denoising
%ct_H = cumsum(dHdx, 2);
%ct_V = cumsum(dVdy, 1);

% The vertical pass is performed using a transposed image.
dVdy = dVdy';

%% Perform the filtering.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = num_iterations;
F = I;

sigma_H = sigma_s;

for i = 0:num_iterations - 1
    
    % Compute the sigma value for this iteration (Equation 14 of our paper).
    sigma_H_i = sigma_H * sqrt(3) * 2^(N - (i + 1)) / sqrt(4^N - 1);
    
    F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i);
    
    
    F = image_transpose(F);
    
    F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i);
    F = image_transpose(F);
    
end

F = cast(F, class(img));

end

%% Recursive filter.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function F = TransformedDomainRecursiveFilter_Horizontal(I, D, sigma)

% Feedback coefficient (Appendix of our paper).
a = exp(-sqrt(2) / sigma);

F = I;
V = a.^D;

[h w num_channels] = size(I);

% Left -> Right filter.
for i = 2:w
    for c = 1:num_channels
        F(:,i,c) = F(:,i,c) + V(:,i) .* ( F(:,i - 1,c) - F(:,i,c) );
    end
end

% Right -> Left filter.
for i = w-1:-1:1
    for c = 1:num_channels
        F(:,i,c) = F(:,i,c) + V(:,i+1) .* ( F(:,i + 1,c) - F(:,i,c) );
    end
end

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function T = image_transpose(I)

[h w num_channels] = size(I);

T = zeros([w h num_channels], class(I));

for c = 1:num_channels
    T(:,:,c) = I(:,:,c)';
end

end
