function [ MoviesOut_struct, SingularVals ] = SVDfilt( MovieIn, Type, Thresholds )
%SVDFILT SVD spatio-temporal filtering of an input movie according to
% predefined thresholds.
%
% Syntax:
% ------------
% [ MoviesOut_struct ] = SVDfilt( MovieIn, Type, Thresholds )
%
% Inputs:
% ------------
% MovieIn    - Input [M, N, K] movie (can either be real or complex).
% Type       - Type of thresholds. Either represents percentage or index number
%              to cut the SVD diagonal according to. Options are:
%              'perc' - percentage from the largest singular value
%              'ind'  - indeices
% Thresholds - A vector containing the cutoff parameters for the SVD.
%              According to Type, if they represent percentage, they should be between
%              (0, 1) (0 and 1 should not be specified). Otherwise they should correspond to the indices to cut from.
%              For example if Thresholds = [0.2 0.6], then three movies will be produced, each
%              corresponds to the 20% first singular values, the 20%-60% of the singular
%              values and the last 60%-100% of the singular values. The singular values
%              order is always decreasing.
%
% Outputs:
% ------------
% MoviesOut_struct - A struct with the filtered movies, divided according to Thresholds.
%
% Ver. 1 - Written by Oren Solomon, Technion I.I.T. 27-09-2017
%

global DISPLAY

%% Initialization
% -------------------------------------------------------------------------
% Determine size of input movie
[M, N, K] = size(MovieIn);

% Length of Thresholds determines the number of output movies
L = length(Thresholds);

% Rearrange movie in the right from
MovieTmp = reshape(MovieIn, [M*N, K]);

PltInds  = [];

%% Perform SVD decomposition
% -------------------------------------------------------------------------
[U, S, V] = svd(MovieTmp);

% Take the diagonal
sd           = diag(S);
SingularVals = sd;

% Create thresholds - add 0 and 1 (or max) at the beginning and end
switch lower(Type)
    case 'perc'
        if max(Thresholds) > 1
            error('SVDfilt: Thresholds should be between [0 1].');
        end
        
        Thresholds = [0; max(sd)*Thresholds(:); max(sd)];
    case 'ind'
        Thresholds = [1; Thresholds(:); K];
    otherwise
        error('SVDfilt: Type not supported.');    
end

% Cut movies according to thresholds
MovInd = 1;
for ii = 2:L + 2
    sd_tmp = sd;
    switch lower(Type)
        case 'perc'
            % Find relevant indices
            ZeroInds         = find(sd_tmp >= Thresholds(ii - 1) & sd_tmp <= Thresholds(ii));
            sd_tmp(ZeroInds) = 0;
            
            % Indices for the plot
            PltInds = [PltInds ZeroInds(end)];
        case 'ind'
            % Find relevant indices
            TmpBinVec                                    = zeros(K, 1);
            TmpBinVec(Thresholds(ii - 1):Thresholds(ii)) = 1;
            sd_tmp                                       = sd_tmp.*TmpBinVec;
            
            % Indices for the plot
            PltInds = [PltInds Thresholds(ii)];
        otherwise
            error('SVDfilt: Type not supported.');    
    end
    
    % Generate new movie
    % reconstruct the singular values map
    Snew                     = zeros(M*N, K);
    Snew(1:K, 1:K)           = diag(sd_tmp);
    MoviesOut_struct{MovInd} = reshape(U*Snew*V', [M, N K]);
    
    MovInd = MovInd + 1;
end

if DISPLAY
   MaxVal = max(10*log10(SingularVals));
   MinVal = min(10*log10(SingularVals));
    
   figure;
   plot(10*log10(SingularVals), 'b-', 'linewidth', 2); hold on;
   xlabel('Frame number'); ylabel('Singular values [dB]');
   for jj = 1:length(PltInds) - 1
      plot(PltInds(jj)*[1 1], [MinVal MaxVal], 'k'); hold on;
   end
   
   axis([0 length(SingularVals) MinVal MaxVal]);
end














