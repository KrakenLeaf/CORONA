function [ L, S, FuncVal, SingStack ] = decompFISTA( D, E, Et, T, Tt, Params )
%DECOMPFISTA Accelerated decomposition of the spatio-temporal data to low-rank + sparse
%
%
% min_{L, S} 1/2||L + S - D||^2_2 + \lambda_L||L||_* + \lambda_S l_1/l_2(S)
%
%
%
%
%

global VERBOSE DEBUG

%% Initialization
% ------------------------------------------------------------------------------
[M, N]   = size(D);
L        = zeros(M, N);
S        = zeros(M, N);
Lprev    = L;
Sprev    = S;

% Determine if to use naive SVD or more implementation of Candes and Becker
if M*N < 100*100, SMALLSCALE = true; else SMALLSCALE = false; end

% Thresholding parameters
lambda_L = Params.lambda_L;
lambda_S = Params.lambda_S;

% Flags
MonotoneFlag = Params.MonotoneFlag;
PositiveFlag = Params.PositiveFlag;

% Lipschitz constant
Lf       = Params.Lf;

% Init
t        = 1;
t_prev   = 1;

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

% Function value accumulation
FuncVal   = [];
SingStack = []; % Only for debug purposes

% if DEBUG; figure(1986); colormap hot; end
if DEBUG == 1; figure('units','normalized','outerposition',[.15 .1 .7 .85]); colormap hot; end
%% Iterations
% ------------------------------------------------------------------------------
ii = 1;
while ii <= Params.IterMax %&& ConvergenceFlag
    if VERBOSE; IterTic = tic; fprintf(['Iteration #' num2str(ii) '/' num2str(Params.IterMax) ', ']); end
    
    % Intermediary variables update
    Zs = S + ((t_prev - 1)/t)*(S - Sprev);  
    Zl = L + ((t_prev - 1)/t)*(L - Lprev);
    
    % Gradient step
    G = grad(D, Zl, Zs, E, Et);
    
    % SVT on gradient step
    Lprev = L;
    if SMALLSCALE
        [L, ValsSum] = SVT(Zl - (1/Lf)*G, lambda_L/Lf);
    else
        [L, ValsSum] = SVT2(Zl - (1/Lf)*G, lambda_L/Lf, M, N);
    end
    
    % Mixed l_1/l_2 soft thresholding
    Sprev = S;
    S = Tt(mix_soft(T(Zs - (1/Lf)*G), lambda_S/Lf));  
    
    % Positivity constraint for S
    if PositiveFlag; S(S < 0) = 0; end
    
    % Parameter updates for next iteration
    t_prev = t;
    t = 0.5*(1 + sqrt(4*t^2 + 1));
    
    % Calculate function value
    FuncVal = [FuncVal CalcFuncVal(D, L, S, E, lambda_L, lambda_S, ValsSum)];
    
    % Monotonicity
    if ii > 1 && MonotoneFlag
       if FuncVal(end - 1) < FuncVal(end)
          if VERBOSE; fprintf('Monotonicity enforced, '); end
          S = Sprev;
          L = Lprev;
          
          % Remove last function value entry and update with the previous one
          FuncVal = [FuncVal(1:end - 1) FuncVal(end - 1)];
       end
    end
    
    % Check for convergence 
%     if norm(L + S - Lprev - Sprev, 2) <= tol*norm(Lprev + Sprev, 2)
%         ConvergenceFlag = 0;
%     else 
        ii = ii + 1; % Continue with next iteration
%     end
    
    % For debug purposes
    if DEBUG == 1; debugShow(L, S, FuncVal, ii, M); end
    
    % Additional debugging - can be very slow
    if DEBUG == 2
        TmpSing = svd(L + S);
        SingStack = [SingStack TmpSing];
    end
    
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
rInc = 20; 4;                                                       % Increase for speed, decrease for accuracy  
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
U = U(:,1:r); V = V(:,1:r); sigma = sigma(1:r) - tau;% Soft thresholding
% U = U(:,1:r); V = V(:,1:r); sigma = sigma(1:r);  % Hard thresholding

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





