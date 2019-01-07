function out = PlugPlayADMM_general(y,A,lambda,method,opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%out = PlugPlayADMM_general(y,A,lambda,method,opts)
%solves the general inverse problem
%
%inversion step: x=argmin_x(||Ax-y||^2+rho/2||x-(v-u)||^2)
%denoising step: v=Denoise(x+u)
%      update u: u=u+(x-v)
%
%Input:           y    -  the observed gray scale image
%                 A    -  forward operator
%              lambda  -  regularization parameter
%              method  -  denoiser, e.g., 'BM3D'
%       opts.rho       -  internal parameter of ADMM {1}
%       opts.gamma     -  parameter for updating rho {1}
%       opts.maxitr    -  maximum number of iterations for ADMM {20}
%       opts.tol       -  tolerance level for residual {1e-4}   
%       ** default values of opts are given in {}. 
%
%Output:          out  -  recovered gray scale image 
%
%Xiran Wang and Stanley Chan
%Copyright 2016
%Purdue University, West Lafayette, In, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check inputs
if nargin<4
    error('not enough input, try again \n');
elseif nargin==4
    opts = [];
end

% Check defaults
if ~isfield(opts,'rho')
    opts.rho = 1;
end
if ~isfield(opts,'max_itr')
    opts.max_itr = 20;
end
if ~isfield(opts,'tol')
    opts.tol = 1e-4;
end
if ~isfield(opts,'gamma')
    opts.gamma=1;
end
if ~isfield(opts,'print')
    opts.print = false;
end


% set parameters
max_itr   = opts.max_itr;
tol       = opts.tol;
gamma     = opts.gamma;
rho       = opts.rho;

%initialize variables
dim         = size(y);          %dimension of the image
N           = dim(1)*dim(2);    %number of pixels in the image
v           = 0.5*ones(dim);
x           = v;
u           = zeros(dim);
residual    = inf;

%set function handle for denoiser
switch method
    case 'BM3D'
        denoise=@wrapper_BM3D;
    case 'TV'
        denoise=@wrapper_TV;
    case 'NLM'
        denoise=@wrapper_NLM;
    case 'RF'
        denoise=@wrapper_RF;
    otherwise
        error('unknown denoiser \n');
end

% main loop

if opts.print==true
    fprintf('Plug-and-Play ADMM --- General \n');
    fprintf('Denoiser = %s \n\n', method);
    fprintf('itr \t ||x-xold|| \t ||v-vold|| \t ||u-uold|| \n');
end

itr = 1;
while(residual>tol&&itr<=max_itr)
    %store x, v, u from previous iteration for psnr residual calculation
    x_old = x;
    v_old = v;
    u_old = u;
    
    G = @(z,trans_flag) gfun(z,trans_flag,A,rho,dim);
    %inversion step
    xtilde = v-u;
    rhs    = [y(:); sqrt(rho)*xtilde(:)];
    [x,~]  = lsqr(G,rhs,1e-3);
    x      = reshape(x,dim);
    
    %denoising step
    vtilde = x+u;
    vtilde = proj(vtilde);
    sigma  = sqrt(lambda/rho);
    v      = denoise(vtilde,sigma);
    
    %update langrangian multiplier
    u      = u + (x-v);
    
    %update rho
    rho = rho*gamma;
    
    %calculate residual
    residualx = (1/sqrt(N))*(sqrt(sum(sum((x-x_old).^2))));
    residualv = (1/sqrt(N))*(sqrt(sum(sum((v-v_old).^2))));
    residualu = (1/sqrt(N))*(sqrt(sum(sum((u-u_old).^2))));
    
    residual = residualx + residualv + residualu;

    if opts.print==true
        fprintf('%3g \t %3.5e \t %3.5e \t %3.5e \n', itr, residualx, residualv, residualu);
    end
    
    itr = itr+1;
end
out = v;
end


function y = gfun(x,transp_flag,A,rho,dim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This gfun assumes that A is an N-by-N matrix
% If A is M-by-N, then the following program can be changed
% to x1 = x(1:M); x2 = (M+1:M+N)
%
% Stanley Chan
% Nov 26, 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rows = dim(1);
cols = dim(2);
N    = rows*cols;
if strcmp(transp_flag,'transp')         % y = A'*x
    x1   = x(1:N);
    x2   = x(N+1:2*N);
    Atx  = A(x1,'transp');
    y    = Atx + sqrt(rho)*x2;
elseif strcmp(transp_flag,'notransp')   % y = A*x
    Ax   = A(x,'notransp');
    y    = [Ax; sqrt(rho)*x];
end
end