% Plot the RMSE results for Exp. 4 RMSE vs \Delta \theta
% Author: Georgios K. Papageorgiou
% Date: 19/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  clear all;
close all;

% Load the results
T = 500; % number of snapshots
ang_sep_vec = [1 2 4 6 10 14]; % SNR values
SNR = -10;
% Load the MUSIC,R-MUSIC, CRLB results
File = fullfile(save_path,'RMSE_K2_vs_ang_sep_T500_min10dBSNR_v2.mat');
load(File);
% Load the CNN results
% filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
%         'DoA_Estimation_underdetermined\MUSIC_RESULTS\RMSE_CNN_K2_vs_ang_sep_T500_min10dBSNR.h5');    
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\RMSE_CNN_K2_vs_ang_sep_T500_min10dBSNR_new_vf2.h5');      
RMSE_CNN = double(h5read(filename, '/CNN_RMSE'));
% Load the l1-SVD results
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1_SVD_16ULA_K2_min10dBSNR_T500_3D_vs_ang_sep_v2.h5');
RMSE_l1SVD = double(h5read(filename2, '/RMSE_l1_SVD'));
% Load the ESPRIT results
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T500_3D_vs_ang_sep_v2.h5');
RMSE_UnESPRIT = double(h5read(filename3, '/RMSE_UnESPRIT'));

% RMSE_MLP = [NaN, NaN, NaN, NaN, 0.39, 0.42]; % T=500
RMSE_MLP = [NaN, NaN, NaN, NaN, 0.29, 0.36]; % T=1000

%% 
% New Color Options
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

f=figure(1);
plot(ang_sep_vec,RMSE_sam,'^--','Color',orange);
hold on;
plot(ang_sep_vec,RMSE_sam_rm,'o-.','Color',gold_yellow);
plot(ang_sep_vec,RMSE_sam_esp,'+--','Color','r');
plot(ang_sep_vec,RMSE_UnESPRIT,'s-.','Color','g');
plot(ang_sep_vec,RMSE_l1SVD,'*--','Color','m');
plot(ang_sep_vec,RMSE_MLP,'x--','Color','c');
plot(ang_sep_vec,RMSE_CNN,'d--','Color','b');
% plot(ang_sep_vec, CRLB,'v-','Color','y');
plot(ang_sep_vec, CRLB_uncr,'.-','Color','k');
hold off;
set(gca, 'YScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','UnESPRIT','$\ell_{2,1}$-SVD','MLP','CNN','CRLB$_{uncr}$',...
    'interpreter','latex');
xticks([0 1 2 4 6 8 10 12 14]);
yticks([1 3 5:5:35]);
title(['DoA-estimation of K=2 sources at ',num2str(SNR), ' dB SNR'], 'interpreter','latex');
ylabel('RMSE [degrees]', 'interpreter','latex');
xlabel('$\Delta \theta$ [degrees]', 'interpreter','latex');
grid on;

% savefig(f,'RMSE_exp2_T100_cnn_res1_fixed_ang_offgrid.fig');
