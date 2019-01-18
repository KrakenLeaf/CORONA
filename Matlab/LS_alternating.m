function [ L, S, FuncVal ] = LS_alternating( D, E, Et, T, Tt, Params )
%LS_ALTERNATING Summary of this function goes here
%   Detailed explanation goes here
%
%
%
%

global VERBOSE DEBUG

%% Initialization
% ------------------------------------------------------------------------------
[K, N] = size(D);
L = zeros(K, N);
S = zeros(K, N);

% Determine if to use naive SVD or more implementation of Candes and Becker
if K*N < 100*100, SMALLSCALE = true; else SMALLSCALE = false; end

% Thresholding parameters
lambda_L = Params.lambda_L;
lambda_S = Params.lambda_S;

% Lipschitz constant
Lf = 1; %Params.Lf;

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

% Initialization of M
M = Et(D);

%% Iterations
% ------------------------------------------------------------------------------
ii = 1;
while ii <= Params.IterMax %&& ConvergenceFlag
    % SVT for low-rank part
    Lprev = L;
    if SMALLSCALE
        [L, ValsSum] = SVT(M - S, lambda_L/Lf);
    else
        [L, ValsSum] = SVT2(M - S, lambda_L/Lf, M, N);
    end
    
    % Soft thresholding for sparse part
    S = Tt(soft_th(T(M - Lprev), lambda_S/Lf));
    
    % Data consistency
    M = L + S  - grad(D, L, S, E, Et);
    
    % Calculate function value
    FuncVal = [FuncVal CalcFuncVal(D, L, S, E, lambda_L, lambda_S, ValsSum)];
    
    % For debug purposes
    if DEBUG; debugShow(L, S, FuncVal, ii, K); end
    
    ii = ii + 1;
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

% Soft thresholding
function y = soft_th(x, lambda)
% x - matrix
% y - matrix
[M, N] = size(x);
z = x(:);
y = reshape( (z./abs(z)).*max(abs(z) - lambda, 0), [M, N] );

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

function debugShow(L, S, FuncVal, ii, M)
subplot('position', [0.01 0.52 0.42 0.42]); % [x y width height]
imagesc(reshape(L(:, 1), sqrt(M)*[1 1]));axis square; colorbar; title(['Iteration #' num2str(ii) ': Low-rank']);
set(gca, 'ytick', []); set(gca, 'xtick', []);
subplot('position', [0.01 0.02 0.42 0.42]); % [x y width height]
imagesc(reshape(S(:, 1), sqrt(M)*[1 1]));axis square; colorbar; title(['Iteration #' num2str(ii) ': Sparse']);
set(gca, 'ytick', []); set(gca, 'xtick', []);
subplot('position', [0.5 0.08 0.42 0.85]); % [x y width height]
semilogy(FuncVal, '-*b', 'linewidth', 2); title(['Cost function value']);grid on;
xlabel('Iteration number'); set(gca, 'fontsize', 14);
drawnow;















