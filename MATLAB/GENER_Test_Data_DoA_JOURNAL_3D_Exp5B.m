% DoA estimation via CNN: Exp. 5B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 20/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
%clc;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location of the data
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA_16ULA_K2_min10dBSNR_T1000_3D_slideang_offgrid_sep4_power_mismatch.h5');
% Location of the l1-SVD results (without saving the y data-RMSE only)
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1SVD_16ULA_K2_min10dBSNR_T1000_3D_slideang_offgrid_sep4_power_mismatch.h5');
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T1000_3D_slideang_offgrid_sep4_power_mismatch.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 1000; % number of snapshots
SNR_dB = -10; % SNR values
SOURCE.K = 2; % number of sources/targets - Kmax
ULA.N = 16;
SOURCE.interval = 60;
Nsim = 116;
res = 1;
ang_sep = 4; % in degrees
% UnESPRIT pars 
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
ms = 8; % if 1 the weights are equal if ms>1 there are higher weights at the center elements of each subarray
w = min(ms,ULA.N-ds-ms+1);  % Eq 9.133 in [1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE.power = ones(1,SOURCE.K).^2;
noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
SOURCE.true_power = [0.7, 1.25];
true_SNR = 10*log10(min(SOURCE.true_power)/noise_power);

THETA_angles = -SOURCE.interval:res:SOURCE.interval;

% Threshold for the l1-SVD
threshold = 400;
l1_SVD_doa_est = zeros(SOURCE.K,Nsim);
UnESPRIT_doa_est = zeros(SOURCE.K,Nsim);
MSE_l1_svd = 0;
MSE_unesp = 0;

theta(1) = -59.43;
theta(2) = theta(1) + ang_sep;

for i=1:Nsim
%theta1 = sort(datasample(Reg,SOURCE.K,'Replace',false));   
A_ula =zeros(ULA.N,SOURCE.K);
%theta = zeros(1,SOURCE.K);
for k=1:SOURCE.K 
    A_ula(:,k) = ULA_steer_vec(theta(k),ULA.N);
end  

% The true covariance matrix 
% Ry_the = A_ula*diag(SOURCE.true_power)*A_ula' + noise_power*eye(ULA.N);
% The signal plus noise
S = sqrt(SOURCE.true_power)'.*(randn(SOURCE.K,T)+1j*randn(SOURCE.K,T))/sqrt(2); 
X = A_ula*S;
Eta = sqrt(noise_power)*(randn(ULA.N,T)+1j*randn(ULA.N,T))/sqrt(2);
Y = X + Eta;

% Calculate the l1-SVD performance without storing the MMV data - Y
[ang_est_l1svd, sp_val_l1svd] = l1_SVD_DoA_est(Y,ULA.N,threshold,SOURCE.K, THETA_angles);
l1_SVD_doa_est(:,i) = sort(ang_est_l1svd);

% Un-ESPRIT results
doas_unit_ESPRIT_sam = unit_ESPRIT(Y, T, ds, SOURCE.K, w);
UnESPRIT_doa_est(:,i) = sort(doas_unit_ESPRIT_sam);

% The sample covariance matrix
Ry_sam = Y*Y'/T;

% % Real and Imaginary part for the sample matrix 
r.sam(:,:,1,i) = real(Ry_sam); 
r.sam(:,:,2,i) = imag(Ry_sam);
r.sam(:,:,3,i) = angle(Ry_sam);

% The angles - Ground Truth
r.angles(:,i) = theta';

MSE_l1_svd = MSE_l1_svd + norm(r.angles(:,i)-l1_SVD_doa_est(:,i))^2;
MSE_unesp = MSE_unesp + norm(r.angles(:,i)-UnESPRIT_doa_est(:,i))^2;

theta = theta+1;
i
end

RMSE_l1_svd = sqrt(MSE_l1_svd/Nsim/SOURCE.K);
RMSE_unesp = sqrt(MSE_unesp/Nsim/SOURCE.K);
time_tot = toc/60; % in minutes

% Save the variables in the specified locations
h5create(filename,'/sam', size(r.sam));
h5write(filename, '/sam', r.sam);
h5create(filename,'/angles',size(r.angles));
h5write(filename, '/angles', r.angles);
h5create(filename2,'/l1_SVD_ang',size(l1_SVD_doa_est));
h5write(filename2, '/l1_SVD_ang', l1_SVD_doa_est);
h5create(filename3,'/UnESPRIT_ang',size(UnESPRIT_doa_est));
h5write(filename3, '/UnESPRIT_ang', UnESPRIT_doa_est);
h5disp(filename3);

