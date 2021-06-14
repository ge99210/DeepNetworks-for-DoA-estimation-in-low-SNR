
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 7/9/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 1000; % number of snapshots
threshold_vec = 350:10:450;
SNR_dB = -10; % SNR values
SOURCE_K = 2; % number of sources/targets - Kmax
ULA_N = 16;
SOURCE.interval = 60;
Nsim = 20;
res = 1;
ang_sep = 3.8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE_power = ones(1,SOURCE_K).^2;
THETA_angles = -SOURCE.interval:res:SOURCE.interval;

% These are the angles 
theta = 10;
theta(2) = theta(1) + ang_sep;
ang_gt = theta';
rho = 0.2;
Pc = [1 0; rho sqrt(1-rho^2)];

A_ula =zeros(ULA_N,SOURCE_K);
for k=1:SOURCE_K 
   A_ula(:,k) = ULA_steer_vec(ang_gt(k),ULA_N);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RMSE_MUSIC = zeros(1,length(threshold_vec));
RMSE_RMUSIC = zeros(1,length(threshold_vec));
RMSE_l1SVD = zeros(1,length(threshold_vec));

noise_power = min(SOURCE_power)*10^(-SNR_dB/10);

parfor ii=1:length(threshold_vec)
    threshold  = threshold_vec(ii);
    
    r_sam = zeros(ULA_N,ULA_N,3,Nsim);
    r_the = zeros(ULA_N,ULA_N,3,Nsim);
    Y_dr = zeros(ULA_N,ULA_N,Nsim);
    
    mse_sam = 0;
    mse_sam_rm = 0;
    mse_l1svd = 0;
    
    for i=1:Nsim
    
     % The signal plus noise
    S = (randn(SOURCE_K,T)+1j*randn(SOURCE_K,T))/sqrt(2); 
    X = A_ula*Pc*S;
    Eta = sqrt(noise_power)*(randn(ULA_N,T)+1j*randn(ULA_N,T))/sqrt(2);
    Y = X + Eta;
    % The sample covariance matrix
    Ry_sam = Y*Y'/T;
    
    % MUSIC estimate
    [doas_sam, spec_sam, specang_sam] = musicdoa(Ry_sam,SOURCE_K, 'ScanAngles', THETA_angles);
    ang_sam  = sort(doas_sam)';  
    % RMSE calculation
    mse_sam = mse_sam + norm(ang_sam- ang_gt)^2;
      
    % Root-MUSIC estimator 
    doas_sam_rm = sort(rootmusicdoa(Ry_sam, SOURCE_K))';
    ang_sam_rm = sort(doas_sam_rm);
    % RMSE calculation - degrees
    mse_sam_rm = mse_sam_rm + norm(ang_sam_rm - ang_gt)^2;
   
    % l1_SVD
    [ang_est_l1svd, sp_val_l1svd] = l1_SVD_DoA_est(Y,ULA_N,threshold,SOURCE_K, THETA_angles);
    mse_l1svd = mse_l1svd + norm(sort(ang_est_l1svd) - ang_gt)^2;
        
    end
   
    RMSE_MUSIC(ii) = sqrt(mse_sam/SOURCE_K/Nsim);
    RMSE_RMUSIC(ii) = sqrt(mse_sam_rm/SOURCE_K/Nsim);
    RMSE_l1SVD(ii) = sqrt(mse_l1svd/SOURCE_K/Nsim);
    
ii
end

time_tot = toc/60; % in minutes

figure(1);
plot(threshold_vec, RMSE_MUSIC);
hold on;
plot(threshold_vec, RMSE_RMUSIC);
plot(threshold_vec, RMSE_l1SVD);
hold off;
legend('MUSIC','R-MUSIC','$\ell_1$-SVD', 'interpreter','latex');


[val,ind] = min(RMSE_l1SVD);
best_thresh = threshold_vec(ind);
disp(['The best threshold value is ',num2str(best_thresh)]);
RMSE_l1SVD(ind)
