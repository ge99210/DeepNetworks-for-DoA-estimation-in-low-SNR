% TESTING DATA Generator 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 19/9/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
rng(2015);
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5');
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1_SVD_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5');
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ang_sep = 3.6;
SNR_dB = -10;
T_vec = 1000*[0.1 0.2 0.5 1 2 5 10]; % SNR values
SOURCE_K = 2; % number of sources/targets - Kmax
ULA_N = 16;
SOURCE.interval = 60;
Nsim = 1000;
% UnESPRIT pars 
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
reweight = 8; % if 1 the weights are equal if reweight>1 there are higher weights at the reweight center elements of each subarray
w = min(reweight,ULA_N-ds-reweight+1);
Ns = ULA_N-ds;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE.power = ones(1,SOURCE_K).^2;
noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
% grid
res = 1;
THETA_angles = -SOURCE.interval:res:SOURCE.interval;
thresh_vec = [130 180 270 410 570 910 1280];

% These are the angles
theta(1) = -13.18;
theta(2) = theta(1) + ang_sep;
    
A_ula =zeros(ULA_N,SOURCE_K);
for k=1:SOURCE_K 
    A_ula(:,k) = ULA_steer_vec(theta(k),ULA_N);
end  

% Initialization
RMSE_l1SVD = zeros(1,length(T_vec));
RMSE_UnESPRIT = zeros(1,length(T_vec));
R_sam = zeros(ULA_N,ULA_N,3,Nsim,length(T_vec));

parfor ii=1:length(T_vec)
    T = T_vec(ii);
    threshold = thresh_vec(ii);
    
    mse_l1svd = 0; 
    mse_unesp = 0; 
    r_sam = zeros(ULA_N,ULA_N,3,Nsim);
    
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
    
%     % l1_SVD exploiting group sparsity 
%     [ang_est_l1svd, sp_val_l1svd] = l1_SVD_DoA_est(Y,ULA_N,threshold,SOURCE_K, THETA_angles);
%     mse_l1svd = mse_l1svd + norm(sort(ang_est_l1svd) - sort(theta)')^2;
%     
    % Un-ESPRIT results
   weights = diag(sqrt([1:w-1 w*ones(1,Ns-2*(w-1)) w-1:-1:1])); % Eq 9.132 in [1]
   O = zeros(Ns,ds);
   % Js1 = [weights O]; % don't really need that
   J2 = [O weights];
   % Calculate the Q matrices
   QNs = Q_mat(Ns);
   QN = Q_mat(ULA_N);
   Q2T = Q_mat(2*T);

   % The real-valued data matrix
   TX = QN'*[Y flip(eye(ULA_N))*conj(Y)*flip(eye(T))]*Q2T;
   % Sample covariance of the real-valued data
   Rx_est = TX*TX'/(2*T);

    doas_unit_ESPRIT_sam = unit_ESPRIT_fast(Rx_est, ds, SOURCE_K, w);
    mse_unesp = mse_unesp + norm(sort(doas_unit_ESPRIT_sam) - sort(theta)')^2;
    
    end
 R_sam(:,:,:,:,ii) = r_sam;   
 RMSE_l1SVD(ii) = sqrt(mse_l1svd/SOURCE_K/Nsim);
 RMSE_UnESPRIT(ii) = sqrt(mse_unesp/SOURCE_K/Nsim);
ii
end

time_tot = toc/60; % in minutes

% figure(1);
% plot(T_vec,RMSE_l1SVD,'*--m');
% hold on;
% plot(T_vec,RMSE_UnESPRIT,'s-.m');
% hold off;
% grid on;
% set(gca, 'YScale', 'log');
% ylabel('RMSE [degrees]', 'interpreter','latex');
% xlabel('T [snapshots] $\times 100$', 'interpreter','latex');
% xticks([1 2 5 10 20 50 100]*100);
% xticklabels([1 2 5 10 20 50 100]);

% Save the data
% h5create(filename,'/sam', size(R_sam));
% h5write(filename, '/sam', R_sam);
% h5create(filename,'/angles',size(theta));
% h5write(filename, '/angles', theta);
% h5create(filename2,'/RMSE_l1_SVD',size(RMSE_l1SVD));
% h5write(filename2, '/RMSE_l1_SVD', RMSE_l1SVD);
h5create(filename3,'/RMSE_UnESPRIT',size(RMSE_UnESPRIT));
h5write(filename3, '/RMSE_UnESPRIT', RMSE_UnESPRIT);
% h5disp(filename3);
