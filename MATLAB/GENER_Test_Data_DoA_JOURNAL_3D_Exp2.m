% TESTING DATA Generator - Experiment 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 19/9/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
%clc;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA1K_16ULA_K2_fixed_offgrid_ang_allSNR_T1000_3D.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 1000; % number of snapshots
SNR_dB_vec = -20:5:30; % SNR values
SOURCE_K = 2; % number of sources/targets - Kmax
ULA_N = 16;
SOURCE.interval = 60;
Nsim = 1e+3;
res = 1;
% UnESPRIT pars 
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
ms = 8; % if 1 the weights are equal if ms>1 there are higher weights at the center elements of each subarray
w = min(ms,ULA_N-ds-ms+1);  % Eq 9.133 in [1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE_power = ones(1,SOURCE_K).^2;
THETA_angles = -SOURCE.interval:res:SOURCE.interval;

% These are the angles with D\theta_min=1 degree
theta(1) = 10.11;
theta(2) = 13.3; 
    
A_ula =zeros(ULA_N,SOURCE_K);
for k=1:SOURCE_K 
   A_ula(:,k) = ULA_steer_vec(theta(k),ULA_N);
end  

R_sam = zeros(ULA_N,ULA_N,3,Nsim,length(SNR_dB_vec));
R_the = zeros(ULA_N,ULA_N,3,Nsim,length(SNR_dB_vec));
RMSE_l1SVD = zeros(1,length(SNR_dB_vec));
RMSE_UnESPRIT = zeros(1,length(SNR_dB_vec));

threshold_vec = [1260 700 400 230 140 100 70 70 60 60 60];

parfor ii=1:length(SNR_dB_vec)
    SNR_dB = SNR_dB_vec(ii);
    noise_power = min(SOURCE_power)*10^(-SNR_dB/10);
    
    r_sam = zeros(ULA_N,ULA_N,3,Nsim);
    r_the = zeros(ULA_N,ULA_N,3,Nsim);
    Y_dr = zeros(ULA_N,ULA_N,Nsim);
    mse_l1svd = 0;
    mse_unesp = 0;
    
    for i=1:Nsim
    
    % The true covariance matrix 
    Ry_the = A_ula*diag(ones(SOURCE_K,1))*A_ula' + noise_power*eye(ULA_N);
    % The signal plus noise
    S = (randn(SOURCE_K,T)+1j*randn(SOURCE_K,T))/sqrt(2); 
    X = A_ula*S;
    Eta = sqrt(noise_power)*(randn(ULA_N,T)+1j*randn(ULA_N,T))/sqrt(2);
    Y = X + Eta;
    % The sample covariance matrix
    Ry_sam = Y*Y'/T;

    % Real and Imaginary part for the sample matrix 
    r_sam(:,:,1,i) = real(Ry_sam); 
    r_sam(:,:,2,i) = imag(Ry_sam);
    r_sam(:,:,3,i) = angle(Ry_sam);
    
    r_the(:,:,1,i) = real(Ry_the); 
    r_the(:,:,2,i) = imag(Ry_the);
    r_the(:,:,3,i) = angle(Ry_the);
       
   % l1_SVD exploiting group sparsity
   threshold = threshold_vec(ii); 
   [ang_est_l1svd, sp_val_l1svd] = l1_SVD_DoA_est(Y,ULA_N,threshold,SOURCE_K, THETA_angles);
   mse_l1svd = mse_l1svd + norm(sort(ang_est_l1svd) - theta')^2;
    
   % Un-ESPRIT results
   doas_unit_ESPRIT_sam = unit_ESPRIT(Y, T, ds, SOURCE_K, w);
   mse_unesp = mse_unesp + norm(sort(doas_unit_ESPRIT_sam) - theta')^2;

    end
    R_sam(:,:,:,:,ii) = r_sam;
    R_the(:,:,:,:,ii) = r_the;
    RMSE_l1SVD(ii) = sqrt(mse_l1svd/SOURCE_K/Nsim);
    RMSE_UnESPRIT(ii) = sqrt(mse_unesp/SOURCE_K/Nsim);
ii
end
angles = theta;

time_tot = toc/60; % in minutes

h5create(filename,'/sam', size(R_sam));
h5write(filename, '/sam', R_sam);
h5create(filename,'/the', size(R_the));
h5write(filename, '/the', R_the);
h5create(filename,'/angles',size(angles));
h5write(filename, '/angles', angles);
h5create(filename,'/RMSE_l1SVD',size(RMSE_l1SVD));
h5write(filename, '/RMSE_l1SVD', RMSE_l1SVD);
h5create(filename,'/RMSE_UnESPRIT',size(RMSE_UnESPRIT));
h5write(filename, '/RMSE_UnESPRIT', RMSE_UnESPRIT);
%h5disp(filename);
