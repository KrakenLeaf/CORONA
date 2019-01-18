clc;
clear;
close all;

global DISPLAY VERBOSE DEBUG
DISPLAY = 1;
VERBOSE = 1;
DEBUG   = 0;

% NOTE   : Its probably best to work on the complex IQ data and NOT on the envelope
% Problem: The dynamic range difference between the low-rank background and the
% sparse blood vessels in huge (several orders of magnitude) - this causes the
% algorithm to not work properly. When they are of the same order of magnitude,
% it works good.

%% Load data
% ------------------------------------------------------------------------------
load('FOLDER NAME HERE'); % Input data should be a mat file with the field data_IQ (IQ analytical CEUS signal + clutter);

% Prepare movie 
MovieIn = data_IQ;
MovieIn = MovieIn(64:64+31, 64:64+31, 1:20);

% Normalize movie values to be between [0-1]
MaxElement = max(abs(MovieIn(:)));
MovieIn    = MovieIn/MaxElement;

[M, N, K]  = size(MovieIn);

% %% SVD filtering
% % ------------------------------------------------------------------------------
Type = 'ind';
Thresholds = [2];
[Movies, SingularVals] = SVDfilt( MovieIn, Type, Thresholds );

%% Run solver
% ------------------------------------------------------------------------------
% Input data
Data = reshape(MovieIn, [M*N, K]);

% Operators
E  = @(x) x; %eye(M*N, M*N);
Et = @(x) x; %[];
T  = @(x) x; %eye(M*N, M*N);
Tt = @(x) x; %[];

% Calculate maximum singular value for the data
% MaxSingVal = lansvd(Data, 1, 'L'); 
MaxSingVal = svds(Data, 1, 'L');

% Parameters
Params.Lf           = 2; 1.5; 1;        %*norm(E);
Params.lambda_L     = 10; %(1e-6)*MaxSingVal; %0.001*MaxSingVal;  % Low rank regularization
Params.lambda_S     = 0.1; %(1e-3)*MaxSingVal; %0.001*MaxSingVal; % Sparsity regularization
Params.IterMax      = 2000; 500;
Params.Tol          = 1e-5;
Params.MonotoneFlag = 0;                % Only for FISTA - enforce monotonicity    
Params.PositiveFlag = 1; % Only for FISTA - enforce positivity for S

% Solver
TotalTime = tic;
% [ L_est, S_est, fVal]       = decompISTA( Data, E, Et, T, Tt, Params );
[ L_est_f, S_est_f, fVal_f] = decompFISTA( Data, E, Et, T, Tt, Params );
toc(TotalTime);

% pm(reshape(10*log(L_est_f), [M, N, K]));axis square;
% S_est_f(S_est_f<0)=0; pm(reshape(10*log(S_est_f), [M, N, K]));axis square;

% Function value
figure;
semilogy(fVal_f, '-*'); legend('FISTA'); grid on;
xlabel('Iteration number');ylabel('Cost function value');
title('Low-Rank + Sparse decomposition');
set(gca, 'fontsize', 16);

figure;
subplot(221);
imagesc(reshape(abs(Movies{1}), [M*N, K]));colormap gray;colorbar;title('L SVD');
subplot(222);
imagesc(reshape(abs(Movies{2}), [M*N, K]));colormap gray;colorbar;title('S SVD');
subplot(223);
imagesc(reshape(abs(L_est_f), [M*N, K]));colormap gray;colorbar;title('L FISTA');
subplot(224);
imagesc(reshape(abs(S_est_f), [M*N, K]));colormap gray;colorbar;title('S FISTA');

