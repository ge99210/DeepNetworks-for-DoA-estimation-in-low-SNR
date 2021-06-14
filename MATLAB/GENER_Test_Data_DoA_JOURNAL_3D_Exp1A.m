% DoA estimation via DNN: Training DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 4/7/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
%clc;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5');
% Location for saving the l1-SVD results (no y data saved-just processed)
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1SVD_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5');
% Location for saving the UnESPRIT results (no y data saved-just processed)
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 2000; % number of snapreshots
SNR_dB = -10; % SNR values
SOURCE.K = 2; % number of sources/targets - Kmax
ULA.N = 16;
SOURCE.interval = 60;
Nsim = 116;
res = 1;
ang_sep = 4.7; % in degrees
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
THETA_angles = -SOURCE.interval:res:SOURCE.interval;

% l1-SVD threshold
threshold = 550;
l1_SVD_doa_est = zeros(SOURCE.K,Nsim);
UnESPRIT_doa_est = zeros(SOURCE.K,Nsim);

theta(1) = - SOURCE.interval;
theta(2) = theta(1) + ang_sep;

for i=1:Nsim
%theta1 = sort(datasample(Reg,SOURCE.K,'Replace',false));   
A_ula =zeros(ULA.N,SOURCE.K);
%theta = zeros(1,SOURCE.K);
for k=1:SOURCE.K 
    A_ula(:,k) = ULA_steer_vec(theta(k),ULA.N);
end  

% The true covariance matrix 
Ry_the = A_ula*diag(ones(SOURCE.K,1))*A_ula' + noise_power*eye(ULA.N);
% The signal plus noise
S = (randn(SOURCE.K,T)+1j*randn(SOURCE.K,T))/sqrt(2); 
X = A_ula*S;
Eta = sqrt(noise_power)*(randn(ULA.N,T)+1j*randn(ULA.N,T))/sqrt(2);
Y = X + Eta;

% Calculate the l1-SVD performance without storing the MMV data - Y
[ang_est_l1svd, sp_val_l1svd] = l1_SVD_DoA_est(Y,ULA.N,threshold,SOURCE.K, THETA_angles);
l1_SVD_doa_est(:,i) = sort(ang_est_l1svd)';

% Un-ESPRIT results
doas_unit_ESPRIT_sam = unit_ESPRIT(Y, T, ds, SOURCE.K, w);
UnESPRIT_doa_est(:,i) = sort(doas_unit_ESPRIT_sam);
% The sample covariance matrix
Ry_sam = Y*Y'/T;

% Real and Imaginary part for the sample matrix 
r.sam(:,:,1,i) = real(Ry_sam); 
r.sam(:,:,2,i) = imag(Ry_sam);
r.sam(:,:,3,i) = angle(Ry_sam);

% Real and Imaginary part for the theor. covariance matrix R
r.the(:,:,1,i) = real(Ry_the); 
r.the(:,:,2,i) = imag(Ry_the);
r.the(:,:,3,i) = angle(Ry_the);

% The angles - Ground Truth
r.angles(:,i) = theta';

theta = theta+1;
i
end

time_tot = toc/60; % in minutes

% Save all variables
h5create(filename,'/sam', size(r.sam));
h5write(filename, '/sam', r.sam);
h5create(filename,'/theor',size(r.the));
h5write(filename, '/theor', r.the);
h5create(filename,'/angles',size(r.angles));
h5write(filename, '/angles', r.angles);
h5create(filename2,'/l1_SVD_ang',size(l1_SVD_doa_est));
h5write(filename2, '/l1_SVD_ang', l1_SVD_doa_est);
h5create(filename3,'/UnESPRIT_ang',size(UnESPRIT_doa_est));
h5write(filename3, '/UnESPRIT_ang', UnESPRIT_doa_est);
%h5disp(filename);
