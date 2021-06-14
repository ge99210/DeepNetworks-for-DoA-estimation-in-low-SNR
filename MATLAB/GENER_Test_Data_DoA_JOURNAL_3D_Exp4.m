% TESTING DATA Generator 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 19/9/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
rng(14); % def 14
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA1K_16ULA_K2_min10dBSNR_T500_3D_vs_ang_sep_v2.h5');
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1_SVD_16ULA_K2_min10dBSNR_T500_3D_vs_ang_sep_v2.h5');
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T500_3D_vs_ang_sep_v2.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 500; % number of snapshots
SNR_dB = -10;
ang_sep_vec = [1 2 4 6 10 14]; % SNR values
SOURCE_K = 2; % number of sources/targets - Kmax
ULA_N = 16;
SOURCE.interval = 60;
Nsim = 10;
res = 1;
% UnESPRIT pars 
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
ms = 8; % if 1 the weights are equal if ms>1 there are higher weights at the center elements of each subarray
w = min(ms,ULA_N-ds-ms+1);  % Eq 9.133 in [1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE.power = ones(1,SOURCE_K).^2;
noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
% grid
THETA_angles = -SOURCE.interval:res:SOURCE.interval;

% l1-SVD threshold parameter
threshold = 290;
% Angles
% theta(1) = - 13.8; 
theta(1) = - 13.8; 
theta_all = [theta(1)*ones(1,length(ang_sep_vec)); theta(1) + ang_sep_vec];

% Initialization
RMSE_l1SVD = zeros(1,length(ang_sep_vec));
RMSE_UnESPRIT = zeros(1,length(ang_sep_vec));
R_sam = zeros(ULA_N,ULA_N,3,Nsim,length(ang_sep_vec));
for ii=1:length(ang_sep_vec)  
 
    A_ula =zeros(ULA_N,SOURCE_K);
    for k=1:SOURCE_K 
        A_ula(:,k) = ULA_steer_vec(theta_all(k,ii),ULA_N);
    end  
    mse_l1svd = 0; 
    mse_unesp = 0;
    r_sam = zeros(ULA_N,ULA_N,3,Nsim);
    
    for i=1:Nsim

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

    % l1_SVD exploiting group sparsity 
   [ang_est_l1svd, sp_val_l1svd] = l1_SVD_DoA_est(Y,ULA_N,threshold,SOURCE_K, THETA_angles);
   mse_l1svd = mse_l1svd + norm(sort(ang_est_l1svd) - theta_all(:,ii))^2;
   
   % Un-ESPRIT results
   doas_unit_ESPRIT_sam = unit_ESPRIT(Y, T, ds, SOURCE_K, w);
   mse_unesp = mse_unesp + norm(sort(doas_unit_ESPRIT_sam) - theta_all(:,ii))^2;
   
    end
 R_sam(:,:,:,:,ii) = r_sam;   
 RMSE_l1SVD(ii) = sqrt(mse_l1svd/SOURCE_K/Nsim);
 RMSE_UnESPRIT(ii) = sqrt(mse_unesp/SOURCE_K/Nsim);
ii
end

time_tot = toc/60; % in minutes

figure(1);
plot(ang_sep_vec,RMSE_l1SVD,'--ob');
hold on;
plot(ang_sep_vec,RMSE_UnESPRIT,'--or');
hold off;

% 
h5create(filename,'/sam', size(R_sam));
h5write(filename, '/sam', R_sam);
h5create(filename,'/angles',size(theta_all));
h5write(filename, '/angles', theta_all);
h5create(filename2,'/RMSE_l1_SVD',size(RMSE_l1SVD));
h5write(filename2, '/RMSE_l1_SVD', RMSE_l1SVD);
h5create(filename3,'/RMSE_UnESPRIT',size(RMSE_UnESPRIT));
h5write(filename3, '/RMSE_UnESPRIT', RMSE_UnESPRIT);
h5disp(filename);
