% DoA estimation via CNN - Exp. 6B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 25/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
tic;
rng(100); % 100 is descent not for K=3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location of the data
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA_16ULA_K3_min10dBSNR_T3000_3D_slideang_offgrid_ang_sep10.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 3000; % number of snapshots
SNR_dB = -10; % SNR values
SOURCE.K = 3; % number of sources/targets - Kmax
ULA.N = 16;
SOURCE.interval = 60;
res = 1;
ang_sep = 10; % in degrees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE.power = ones(1,SOURCE.K).^2;
noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
THETA_angles = -SOURCE.interval:res:SOURCE.interval;

% % Threshold for the l1-SVD
% threshold = 60;
% l1_SVD_doa_est = zeros(SOURCE.K,Nsim);
% MSE_l1_svd = 0;

if SOURCE.K == 1 || SOURCE.K == 2 || SOURCE.K == 3
    theta(1) = -59.8;
    Nsim = 120;
end
if SOURCE.K == 2 || SOURCE.K == 3
    theta(2) = theta(1) + ang_sep;
    Nsim = 110;
end
if SOURCE.K == 3
    theta(3) = theta(2) + ang_sep;
    Nsim = 100;
end

for i=1:Nsim
%theta1 = sort(datasample(Reg,SOURCE.K,'Replace',false));   
A_ula =zeros(ULA.N,SOURCE.K);
%theta = zeros(1,SOURCE.K);
for k=1:SOURCE.K 
    A_ula(:,k) = ULA_steer_vec(theta(k),ULA.N);
end  

% The true covariance matrix 
Ry_the = A_ula*diag(SOURCE.power)*A_ula' + noise_power*eye(ULA.N);
% The signal plus noise
S = (randn(SOURCE.K,T)+1j*randn(SOURCE.K,T))/sqrt(2); 
X = A_ula*S;
Eta = sqrt(noise_power)*(randn(ULA.N,T)+1j*randn(ULA.N,T))/sqrt(2);
Y = X + Eta;

% Calculate the l1-SVD performance without storing the MMV data - Y
% [ang_est_l1svd, sp_val_l1svd] = l1_SVD_DoA_est(Y,ULA.N,threshold,SOURCE.K, THETA_angles);
% l1_SVD_doa_est(:,i) = sort(ang_est_l1svd);

% The sample covariance matrix
Ry_sam = Y*Y'/T;

% % Real and Imaginary part for the sample matrix 
r.sam(:,:,1,i) = real(Ry_sam); 
r.sam(:,:,2,i) = imag(Ry_sam);
r.sam(:,:,3,i) = angle(Ry_sam);

% % Real and Imaginary part for the sample matrix 
r.the(:,:,1,i) = real(Ry_the); 
r.the(:,:,2,i) = imag(Ry_the);
r.the(:,:,3,i) = angle(Ry_the);

% MSE_l1_svd = MSE_l1_svd + norm(r.angles(:,i)-l1_SVD_doa_est(:,i))^2;
r.angles(:,i) = theta';

theta = theta +1;
i
end

% RMSE_l1_svd = sqrt(MSE_l1_svd/Nsim/SOURCE.K);
time_tot = toc/60; % in minutes

% Save the variables in the specified locations
h5create(filename,'/sam', size(r.sam));
h5write(filename, '/sam', r.sam);
h5create(filename,'/the', size(r.the));
h5write(filename, '/the', r.the);
h5create(filename,'/angles',size(r.angles));
h5write(filename, '/angles', r.angles);
% h5create(filename2,'/l1_SVD_ang',size(l1_SVD_doa_est));
% h5write(filename2, '/l1_SVD_ang', l1_SVD_doa_est);
% h5disp(filename);
