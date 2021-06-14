% Plot the RMSE results for Experiment 6 
% Author: Georgios K. Papageorgiou
% Date: 08/04/2021 - RQ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all;
close all;
% Load the results
T = 1000; 
rho_vec = 0:0.1:1;
% Load the MUSIC,R-MUSIC, CRLB results
File = fullfile(save_path,'RMSE_K2_offgrid_ang_fixed_min10dBSNR_T1000_vsrho.mat');
load(File);

% Load the CNN results
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\RMSE_CNN_K2_fixed_offgrid_ang3coma8_min10dBSNR_T1000_vsrho_lrRoP_0_7.h5');    
RMSE_CNN = double(h5read(filename, '/CNN_RMSE'));
    
% Load the l1-SVD results
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA1K_3D_16ULA_K2_fixed_ang_sep3coma8_min10dBSNR_T1000_vsrho.h5');  
RMSE_l1_SVD = double(h5read(filename2, '/RMSE_l1SVD'));
RMSE_UnESPRIT = double(h5read(filename2, '/RMSE_UnESPRIT'));

%%
% New Color Options
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

f=figure(1);
plot(rho_vec,RMSE_sam,'^--','Color',orange);
hold on;
plot(rho_vec,RMSE_sam_rm,'o-.','Color',	gold_yellow);
plot(rho_vec,RMSE_sam_esp,'+--','Color','r');
plot(rho_vec,RMSE_UnESPRIT,'s-.','Color','g');
plot(rho_vec,RMSE_l1_SVD,'*--','Color','m');
% plot(SNR_vec,RMSE_MLP,'x--','Color','c');
plot(rho_vec,RMSE_CNN,'d--','Color','b');
plot(rho_vec, CRLB,'.-','Color','k');
hold off;
set(gca, 'YScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','UnESPRIT','$\ell_{2,1}$-SVD','CNN','CRLB',...
    'interpreter','latex','location','northwest');
title(['DoA-estimation of K=2 sources from $T=$',num2str(T), ' snapshots'], 'interpreter','latex');
ylabel('RMSE [degrees]', 'interpreter','latex');
xlabel('Correlation coefficient $\rho$', 'interpreter','latex');
grid on;
