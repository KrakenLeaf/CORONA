function [ L, S, FuncVal ] = decompISTA( D, E, Et, T, Tt, Params )
%DECOMPISTA decomposes the spatio-temporal data to low-rank + sparse
%
%
% min_{L, S} 1/2||L + S - D||^2_2 + \lambda_L||L||_* + \lambda_S l_1/l_2(S)
%
%
%
%
%

global VERBOSE

%% Initialization
% ------------------------------------------------------------------------------
[M, N] = size(D);
L = zeros(M, N);
S = zeros(M, N);

% Determine if to use naive SVD or more implementation of Candes and Becker
if M*N < 100*100, SMALLSCALE = true; else SMALLSCALE = false; end

% Thresholding parameters
lambda_L = Params.lambda_L;
lambda_S = Params.lambda_S;

% Lipschitz constant
Lf = Params.Lf;

% Flag which indicates if solution converged below Params.Tol
ConvergenceFlag = 1;
tol             = Params.Tol;

% Check to see if E and T are operators or not
if isnumeric(E)
   Et = @(x) E'*x;
   E  = @(x) E*x;
end
if isnumeric(T)
   Tt = @(x) T\x; % T^{-1}*x
   T  = @(x) T*x;
end

FuncVal = [];

%% Iterations
% ------------------------------------------------------------------------------
ii = 1;
while ii <= Params.IterMax %&& ConvergenceFlag
    if VERBOSE; IterTic = tic; fprintf(['Iteration #' num2str(ii) '/' num2str(Params.IterMax) ', ']); end
    
    % Store previous iteration
    Lprev = L;
    Sprev = S;

    % Gradient step
    G = grad(D, L, S, E, Et);
    
    % SVT on gradient step
    if SMALLSCALE
        [L, ValsSum] = SVT(L - (1/Lf)*G, lambda_L/Lf);
    else
        [L, ValsSum] = SVT2(L - (1/Lf)*G, lambda_L/Lf, M, N);
    end
    
    % Mixed l_1/l_2 soft thresholding
    S = Tt(mix_soft(T(S - (1/Lf)*G), lambda_S/Lf));  
    
    % Calculate function value
    FuncVal = [FuncVal CalcFuncVal(D, L, S, E, lambda_L, lambda_S, ValsSum)];
    
    % Check for convergence 
%     if norm(L + S - Lprev - Sprev, 2) <= tol*norm(Lprev + Sprev, 2)
%         ConvergenceFlag = 0;
%     else 
        ii = ii + 1; % Continue with next iteration
%     end
    
    if VERBOSE; disp([' time = ' num2str(toc(IterTic)*1000) 'ms.']); end
end


%% Auxiliary functions
% ------------------------------------------------------------------------------
% Singular value thresholding - naive implementation
function [Y, ValsSum] = SVT(X, lambda)
% X - matrix
% Y - matrix
[U, S, V]                       = svd(X);
Tmp                             = zeros(size(X));
ThresholdedVals                 = soft(diag(S), lambda);
Tmp(1:size(S, 2), 1:size(S, 2)) = diag(ThresholdedVals); 
Y                               = U*Tmp*V';

% For calculating the cost function easily
ValsSum                         = sum(ThresholdedVals);

% Singular value thresholding - more efficient method
function [Y, ValsSum] = SVT2(X, tau, M, N)
% This code is editted from the SVT code of Emmanuel Candes & Stephen Becker, March 2009
% How many singular values to compute
rInc = 4;                                                       % Increase for speed, decrease for accuracy
s    = min( [rInc, M, N] );

% Calculate only a few singular values - inside of a loop
OK = 0;
while ~OK 
%     [U, Sigma, V] = lansvd(X, s, 'L');                          % Compute SVD
    [U, Sigma, V] = svds(X, s, 'L');                            % Compute SVD
    OK            = (Sigma(s,s) <= tau) || ( s == min(M, N) );  % Loop stopping criterion
    s             = min(s + rInc, min(M, N));                   % Increase number of calculated singular values
end

% Soft-thresholding of singular values
sigma = diag(Sigma); r = sum(sigma > tau);
U = U(:,1:r); V = V(:,1:r); sigma = sigma(1:r) - tau; Sigma = diag(sigma);

% Recompose output
Y       = U*diag(sigma)*V';
ValsSum = sum(sigma);

% Soft thresholding
function y = soft(x, lambda)
% x - vector
% y - vector
y = (x./abs(x)).*max(abs(x) - lambda, 0);

% Mixed l_1/l_2 soft thresholding
function Y = mix_soft(X, lambda)
% Calculate the l_2 norm for each row
X_nrm = sqrt(sum(abs(X).^2, 2));
temp  = max(0, 1 - lambda./X_nrm);
Y     = X.*repmat(temp, [1 size(X, 2)]); 

% Similar part of both gradients
function G = grad(D, L, S, E, Et)
G = Et(E(L + S) - D);

% Calculate function value for each iteration
function Fval = CalcFuncVal(D, L, S, E, lambda_L, lambda_S, ValsSum)
QuadTerm = 0.5*norm(E(L + S) - D, 2)^2;                % Quadratic term
LTerm    = ValsSum; %sum(svd(L));                      % Nuclear Norm
STerm    = sum(sqrt(sum(abs(S).^2, 2)));               % Mixed l1/l2 norm
Fval     = QuadTerm + lambda_L*LTerm + lambda_S*STerm; % Cost function value








