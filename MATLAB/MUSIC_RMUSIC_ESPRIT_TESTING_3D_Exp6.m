% CNN Testing - Experiment 2
% Georgios K. Papageorgiou 19/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA1K_3D_16ULA_K2_fixed_ang_sep3coma8_min10dBSNR_T1000_vsrho.h5');
%h5disp(filename);
r_sam = h5read(filename, '/sam');
R_sam = squeeze(r_sam(:,:,1,:,:)+1j*r_sam(:,:,2,:,:));
r_the = h5read(filename, '/the');
R_the = squeeze(r_the(:,:,1,:,:)+1j*r_the(:,:,2,:,:));
True_angles = h5read(filename, '/angles');
SOURCE_K = size(True_angles,2);
[ULA_N,~, N_test,rhos] = size(R_sam);
rho_vec = 0:0.1:1;
SOURCE_power = ones(1, SOURCE_K);
SOURCE.interval = 60;
res = 1;
SNR_dB = -10;
T = 1000;
noise_power = min(SOURCE_power)*10^(-SNR_dB/10);
% UnESPRIT pars 
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
ms = 8; % if 1 the weights are equal if ms>1 there are higher weights at the center elements of each subarray
w = min(ms,ULA_N-ds-ms+1);  % Eq 9.133 in [1] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For the CRLB
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
der_a = @(x,N) 1j*(pi^2/180)*cos(deg2rad(x))*ULA_steer_vec(x,N).*(0:1:N-1)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_ula = zeros(ULA_N,SOURCE_K);
D = zeros(ULA_N,SOURCE_K);
for k=1:SOURCE_K  
    A_ula(:,k) = ULA_steer_vec(True_angles(k),ULA_N);
    D(:,k) = der_a(True_angles(k),ULA_N);
end
H = D'*(eye(ULA_N)-A_ula*pinv(A_ula))*D;

% Initialization
RMSE_the = zeros(1,rhos);
RMSE_sam = zeros(1,rhos);
RMSE_the_rm = zeros(1,rhos);
RMSE_sam_rm = zeros(1,rhos);
RMSE_the_esp = zeros(1,rhos);
RMSE_sam_esp = zeros(1,rhos);
CRLB = zeros(1,rhos);

for s=1:rhos
    
rho = rho_vec(s);
P = [1 rho; rho 1];
R = A_ula*P*A_ula' + noise_power*eye(ULA_N);
R_inv = inv(R);

    
rmse_the = 0;
rmse_sam = 0;
rmse_the_rm = 0;
rmse_sam_rm = 0;
rmse_the_esp = 0;
rmse_sam_esp = 0;

for nit=1:N_test
   
   % The true covariance matrix
   Rx = R_the(:,:,nit,s);
    
   % The smoothed sample covariance matrix
   Rx_sam = R_sam(:,:,nit,s);
    
   % MUSIC estimator 
   [doas_the, spec_the, specang_the] = musicdoa(Rx,SOURCE_K,'ScanAngles', -SOURCE.interval:res:SOURCE.interval);
   [doas_sam, spec_sam, specang_sam] = musicdoa(Rx_sam,SOURCE_K, 'ScanAngles', -SOURCE.interval:res:SOURCE.interval);
  
   ang_the = sort(doas_the)';
   ang_sam  = sort(doas_sam)';
   ang_gt = sort(True_angles)';
   
   % RMSE calculation
   rmse_the = rmse_the + norm(ang_the - ang_gt)^2;
   rmse_sam = rmse_sam + norm(ang_sam- ang_gt)^2;
      
   % Root-MUSIC estimator 
   doas_the_rm = sort(rootmusicdoa(Rx, SOURCE_K))';
   doas_sam_rm = sort(rootmusicdoa(Rx_sam, SOURCE_K))';

   ang_the_rm= sort(doas_the_rm);
   ang_sam_rm = sort(doas_sam_rm);
   
   % RMSE calculation - degrees
   rmse_the_rm = rmse_the_rm + norm(ang_the_rm - ang_gt)^2;
   rmse_sam_rm = rmse_sam_rm + norm(ang_sam_rm - ang_gt)^2;
   
   %% ESPRIT (with variable ds and reweighting technique)   
   % EPSRIT
   doas_the_esp = ESPRIT_doa(Rx, ds, SOURCE_K, w);
   doas_sam_esp = ESPRIT_doa(Rx_sam, ds, SOURCE_K, w);
     
   ang_the_esp = sort(doas_the_esp);
   ang_sam_esp = sort(doas_sam_esp);
   
   % ang = espritdoa(Rx_sam,SOURCE_K);
   rmse_the_esp = rmse_the_esp + norm(ang_the_esp - ang_gt)^2;
   rmse_sam_esp = rmse_sam_esp + norm( ang_sam_esp- ang_gt)^2;
   
end

% MUSIC RMSE_deg
RMSE_the(s) = sqrt(rmse_the/SOURCE_K/N_test);
RMSE_sam(s) = sqrt(rmse_sam/SOURCE_K/N_test);

% R-MUSIC RMSE_deg
RMSE_the_rm(s) = sqrt(rmse_the_rm/SOURCE_K/N_test);
RMSE_sam_rm(s) = sqrt(rmse_sam_rm/SOURCE_K/N_test);

% ESPRIT RMSE_deg
RMSE_the_esp(s) = sqrt(rmse_the_esp/SOURCE_K/N_test);
RMSE_sam_esp(s) = sqrt(rmse_sam_esp/SOURCE_K/N_test);

% Cramer-Rao lower bound
C_Cr = (noise_power/(2*T))*inv(real(H.*(P*A_ula'*R_inv*A_ula*P).'));
CRLB(s) = sqrt(trace(C_Cr)/SOURCE_K);

s
end
%%

figure(1);
plot(rho_vec,RMSE_sam,'^--g');
hold on;
plot(rho_vec,RMSE_sam_rm,'o--c');
plot(rho_vec,RMSE_sam_esp,'+--r')
plot(rho_vec, CRLB,'.-k');
hold off;
set(gca, 'YScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','CRLB',...
    'interpreter','latex');
title('DoA-estimation of K=2 sources', 'interpreter','latex');
ylabel('RMSE [degrees]', 'interpreter','latex');
xlabel('Correlation coefficient', 'interpreter','latex');
grid on;

% Save the results 
save_path = 'C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation\DoA_Estimation_underdetermined\ComparisonRESULTS';
save(fullfile(save_path,'RMSE_K2_offgrid_ang_fixed_min10dBSNR_T1000_vsrho.mat'),'rho_vec','RMSE_sam','RMSE_sam_rm','RMSE_sam_esp','CRLB');
