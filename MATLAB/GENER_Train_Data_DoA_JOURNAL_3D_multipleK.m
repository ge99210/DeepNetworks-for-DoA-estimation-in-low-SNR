% DoA estimation via CNN: Training DATA generation 6A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 18/9/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the DATA
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TRAIN_DATA_16ULA_K1to3_min15dBSNR_res1_3D.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SNR_dB_vec = -20:5:0; % SNR values
SOURCE_K_max = 3; % number of sources/targets - Kmax
ULA_N = 16;
SOURCE.interval = 60;
G_res = 1; % degrees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the N-element ULA
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The training sets of angles 
grid = -SOURCE.interval:G_res:SOURCE.interval;
D = [];
for k=SOURCE_K_max:-1:1
    ang_d0 = combnk(grid,k);
    ang_d = [ang_d0 NaN(length(ang_d0),SOURCE_K_max-size(ang_d0,2))];
    D = [D; ang_d];
end

L = size(D,1);
R_the = zeros(ULA_N, ULA_N,3,L,length(SNR_dB_vec));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Progress bar - comment while debugging
% pbar=waitbar(0,'Please wait...','Name','Progress');
for i =1:length(SNR_dB_vec)
SNR_dB = SNR_dB_vec(i);
noise_power = 10^(-SNR_dB/10);
% Angle selection
r_the = zeros(ULA_N, ULA_N,3,L);
for ii=1:L
    SOURCE_angles = D(ii,:);
    SOURCE_angles = SOURCE_angles(~isnan(SOURCE_angles));
    SOURCE_K = size(SOURCE_angles,2);
    A_ula = zeros(ULA_N,SOURCE_K);
    for k=1:SOURCE_K
        A_ula(:,k) = ULA_steer_vec(SOURCE_angles(k),ULA_N);
    end
% The true covariance matrix 
Ry_the = A_ula*diag(ones(SOURCE_K,1))*A_ula' + noise_power*eye(ULA_N);

% Real and Imaginary part for the theor. covariance matrix R
r_the(:,:,1,ii) = real(Ry_the); 
r_the(:,:,2,ii) = imag(Ry_the);
r_the(:,:,3,ii) = angle(Ry_the);
end
R_the(:,:,:,:,i) = r_the;
i
end
% The angles - Ground Truth
angles = D;

% close(pbar);
time_tot = toc/60; % in minutes

% Save the DATA
h5create(filename,'/theor',size(R_the));
h5write(filename, '/theor', R_the);
h5create(filename,'/angles',size(angles));
h5write(filename, '/angles', angles);
h5disp(filename);
