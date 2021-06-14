% MUSIC+R-MUSIC Testing 
% Georgios K. Papageorgiou 19/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
% clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA1K_16ULA_K2_min10dBSNR_T500_3D_vs_ang_sep_v2.h5');
T = 500;
% h5disp(filename);
r_sam = h5read(filename, '/sam');
R_sam = squeeze(r_sam(:,:,1,:,:)+1j*r_sam(:,:,2,:,:));
% r_the = h5read(filename, '/the');
% R_the = squeeze(r_the(:,:,1,:,:)+1j*r_the(:,:,2,:,:));
True_angles = h5read(filename, '/angles');
SOURCE_K = size(True_angles,1);
[ULA_N,~, N_test,ang_sep_vec_size] = size(R_sam);
SOURCE.interval = 60;
SOURCE_power = ones(1, SOURCE_K);
res = 1;
SNR_dB = -10;
noise_power = 10^(-SNR_dB/10);
ang_sep_vec = [1 2 4 6 10 14];
% ESPIRT pars
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
ms = 8; % if 1 the weights are equal if ms>1 there are higher weights at the center elements of each subarray
w = min(ms,ULA_N-ds-ms+1);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For the CRLB
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
der_a = @(x,N) 1j*(pi^2/180)*cos(deg2rad(x))*ULA_steer_vec(x,N).*(0:1:N-1)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
THETA_angles = -SOURCE.interval:res:SOURCE.interval;

% Initialization
RMSE_the = zeros(1,ang_sep_vec_size);
RMSE_sam = zeros(1,ang_sep_vec_size);
RMSE_the_rm = zeros(1,ang_sep_vec_size);
RMSE_sam_rm = zeros(1,ang_sep_vec_size);
RMSE_sam_esp = zeros(1,ang_sep_vec_size);
CRLB = zeros(1,ang_sep_vec_size);
CRLB_uncr = zeros(1,ang_sep_vec_size);

for s=1:ang_sep_vec_size

rmse_the = 0;
rmse_sam = 0;
rmse_the_rm = 0;
rmse_sam_rm = 0;
rmse_sam_esp = 0;

theta = True_angles(:,s);
A_ula = zeros(ULA_N,SOURCE_K);
D = zeros(ULA_N,SOURCE_K);
B = zeros(ULA_N^2,SOURCE_K);
D_uncr = zeros(ULA_N^2,SOURCE_K);
for k=1:SOURCE_K  
    A_ula(:,k) = ULA_steer_vec(theta(k),ULA_N);
    D(:,k) = der_a(theta(k),ULA_N);
    B(:,k) = kron(conj(A_ula(:,k)), A_ula(:,k));
    D_uncr(:,k) = kron(conj(D(:,k)), A_ula(:,k)) + kron(conj(A_ula(:,k)), D(:,k));
end
H = D'*(eye(ULA_N)-A_ula*pinv(A_ula))*D;
R = A_ula*A_ula' + noise_power*eye(ULA_N);
R_inv = inv(R);
% These are used in the CRB for uncorrelated sources
PI_A = A_ula*pinv(A_ula);
G = null(B');

for nit=1:N_test
    
   Rx_sam = R_sam(:,:,nit,s);
   % MUSIC estimator 
%    [doas_the, spec_the, specang_the] = musicdoa(Rx,SOURCE_K,'ScanAngles', THETA_angles);
   [doas_sam, spec_sam, specang_sam] = musicdoa(Rx_sam,SOURCE_K, 'ScanAngles', THETA_angles);
%    ang_the = sort(doas_the)';
   ang_sam  = sort(doas_sam)';
   ang_gt = sort(theta);
 
   % RMSE calculation
%    rmse_the = rmse_the + norm(ang_the - ang_gt)^2;
   rmse_sam = rmse_sam + norm(ang_sam- ang_gt)^2;
   % Root-MUSIC estimator 
%    doas_the_rm = sort(rootmusicdoa(Rx, SOURCE_K))';
   doas_sam_rm = rootmusicdoa(Rx_sam, SOURCE_K)';
%    ang_the_rm= sort(doas_the_rm);
   ang_sam_rm = sort(doas_sam_rm);
   % RMSE calculation - degrees
%    rmse_the_rm = rmse_the_rm + norm(ang_the_rm - ang_gt)^2;
   rmse_sam_rm = rmse_sam_rm + norm(ang_sam_rm - ang_gt)^2;
    % EPSRIT
%    doas_the_esp = ESPRIT_doa(Rx, ds, SOURCE_K, w);
   doas_sam_esp = ESPRIT_doa(Rx_sam, ds, SOURCE_K, w);
%    ang_the_esp = sort(doas_the_esp);
   ang_sam_esp = sort(doas_sam_esp);
   % ang = espritdoa(Rx_sam,SOURCE_K);
%    rmse_the_esp = rmse_the_esp + norm(ang_the_esp - ang_gt)^2;
   rmse_sam_esp = rmse_sam_esp + norm(ang_sam_esp- ang_gt)^2;
   
end
% CRB 
C_Cr = trace((noise_power/(2*T))*inv(real(H.*(A_ula'*R_inv*A_ula).')));
% CRB for uncorrelated sources
C = kron(R.', R) + (noise_power^2/(ULA_N-SOURCE_K))*(PI_A(:)*(PI_A(:))');
CRB_mat = inv(diag(SOURCE_power)*D_uncr'*G*inv(G'*C*G)*G'*D_uncr*diag(SOURCE_power))/T;
CRB_uncr = real(trace(CRB_mat));

% MUSIC RMSE_deg
RMSE_the(s) = sqrt(rmse_the/SOURCE_K/N_test);
RMSE_sam(s) = sqrt(rmse_sam/SOURCE_K/N_test);

% R-MUSIC RMSE_deg
RMSE_the_rm(s) = sqrt(rmse_the_rm/SOURCE_K/N_test);
RMSE_sam_rm(s) = sqrt(rmse_sam_rm/SOURCE_K/N_test);

% ESPRIT RMSE deg
% RMSE_the_esp(s) = sqrt(rmse_the_esp/SOURCE_K/N_test);
RMSE_sam_esp(s) = sqrt(rmse_sam_esp/SOURCE_K/N_test);

% CRLB per Delta theta
CRLB(s) = sqrt(C_Cr/SOURCE_K);
CRLB_uncr(s) = sqrt(CRB_uncr/SOURCE_K);

s
end

figure(1);
plot(ang_sep_vec,RMSE_sam,'s--r');
hold on;
plot(ang_sep_vec,RMSE_sam_rm,'o--g');
plot(ang_sep_vec,RMSE_sam_esp,'+--b');
plot(ang_sep_vec, CRLB,'*-k');
plot(ang_sep_vec, CRLB_uncr,'.-');
hold off;
set(gca, 'YScale', 'log');
legend('MUSIC ', 'R-MUSIC ','ESPRIT','CRLB','CRLB$_{uncr}$',...
    'interpreter','latex');
title('DoA-estimation of K=2 sources', 'interpreter','latex');
ylabel('RMSE [degrees]', 'interpreter','latex');
xlabel('Angle separation [degrees]', 'interpreter','latex');
grid on;

% % % %% Save the results 
% save_path = 'C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation\DoA_Estimation_underdetermined\ComparisonRESULTS';
% save(fullfile(save_path,'RMSE_K2_vs_ang_sep_T500_min10dBSNR_v2.mat'),'ang_sep_vec','RMSE_sam','RMSE_sam_rm','RMSE_sam_esp','CRLB','CRLB_uncr');
