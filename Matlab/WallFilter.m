clc;
clear;
close all;



%% Parameters
% -------------------------------------------------------------------------
Folder  = 'mouse2';
MatFile = 'toOren_IQ_bf_data4';

DR = 45; %dB

% Butterworth parameters
PRF    = 100; % Hz
ftype  = 'high';
Cutoff = 0.9; 0.03;
Order  = 6; 

% Filter 
% -------------------------------------------------------------------------
% Load data
load(fullfile(Folder, MatFile));

% Generate Butterworth filter
[b, a] = butter(Order, Cutoff, ftype);

% Perform filtering
IQ_f = filter(b, a, data_IQ, [], 3);

%% Display MIP
% -------------------------------------------------------------------------
MIP = max(abs(IQ_f), [], 3);

MIP = MIP + abs(min(MIP(:)));    
MIP = db(MIP/max(max(MIP(:))));

MIP = MIP(50:120, 50:200);

figure; colormap gray
imagesc(MIP, [-DR 0]); 

% Save result
save(['Butter_' MatFile '_order' num2str(Order) '_cutoff' num2str(Cutoff*1000) '_divideby1000'], 'IQ_f');



