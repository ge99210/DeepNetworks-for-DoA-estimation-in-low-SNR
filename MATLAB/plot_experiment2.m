% Plot the RMSE results for Experiment 2 with Delta \theta =3.19 vs SNR
% Author: Georgios K. Papageorgiou
% Date: 19/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all;
close all;
% Load the results
T = 1000; 

% Load the MUSIC,R-MUSIC, CRLB results
File = fullfile(save_path,'RMSE_K2_offgrid_ang_fixed_allSNR_T1000.mat');
load(File);

% Load the CNN results
% filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
%         'DoA_Estimation_underdetermined\MUSIC_RESULTS\RMSE_CNN_K2_fixed_offgrid_ang_all_SNR_T1000_new_train_low.h5');    
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\RMSE_CNN_K2_fixed_offgrid_ang_all_SNR_T1000_RQ.h5');    
RMSE_CNN = double(h5read(filename, '/CNN_RMSE'));
    
% Load the l1-SVD results
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA1K_16ULA_K2_fixed_offgrid_ang_allSNR_T1000_3D.h5');  
RMSE_l1_SVD = double(h5read(filename2, '/RMSE_l1SVD'));
RMSE_UnESPRIT = double(h5read(filename2, '/RMSE_UnESPRIT'));

RMSE_MLP = [NaN NaN NaN 0.19 0.17 0.16 0.16 0.16 0.16 0.16 0.16];

%%
% New Color Options
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

f=figure(1);
plot(SNR_vec,RMSE_sam,'^--','Color',orange);
hold on;
plot(SNR_vec,RMSE_sam_rm,'o-.','Color',	gold_yellow);
plot(SNR_vec,RMSE_sam_esp,'+--','Color','r');
plot(SNR_vec,RMSE_UnESPRIT,'s-.','Color','g');
plot(SNR_vec,RMSE_l1_SVD,'*--','Color','m');
plot(SNR_vec,RMSE_MLP,'x--','Color','c');
plot(SNR_vec,RMSE_CNN,'d--','Color','b');
plot(SNR_vec, CRB_uncr,'.-','Color','k');
hold off;
set(gca, 'YScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','UnESPRIT','$\ell_{2,1}$-SVD','MLP','CNN','CRLB$_{uncr}$',...
    'interpreter','latex');
title(['DoA-estimation of K=2 sources from $T=$',num2str(T), ' snapshots'], 'interpreter','latex');
ylabel('RMSE [degrees]', 'interpreter','latex');
xlabel('SNR [dB]', 'interpreter','latex');
grid on;
