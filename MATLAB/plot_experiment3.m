% Plot the RMSE results for Exp. 3 RMSE vs T
% Author: Georgios K. Papageorgiou
% Date: 19/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all;

% Load the results
T_vec = 1000*[0.1 0.2 0.5 1 2 5 10];
SNR = -10;
% Load the MUSIC,R-MUSIC, CRLB results
File = fullfile(save_path,'RMSE_K2_min10dB_ang_sep3coma6_vsT.mat');
load(File);
% Load the CNN results
% filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
%         'DoA_Estimation_underdetermined\MUSIC_RESULTS\RMSE_CNN_K2_min10dBSNR_vsT_ang_sep3coma6_new_train_low.h5');    
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\RMSE_CNN_K2_min10dBSNR_vsT_ang_sep3coma6_new_train_low_vf.h5');    
RMSE_CNN = double(h5read(filename, '/CNN_RMSE'));
% Load the l1-SVD results
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1_SVD_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5');
RMSE_l1SVD = double(h5read(filename2, '/RMSE_l1_SVD'));
% UnESPRIT
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5');
RMSE_UnESPRIT = double(h5read(filename3, '/RMSE_UnESPRIT'));

RMSE_MLP = [1.61, 1.48, 0.83, 0.57, 0.67, 0.18, 0.21];
%%
% New Color Options
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

f=figure(1);
plot(T_vec,RMSE_sam,'^--','Color',orange);
hold on;
plot(T_vec,RMSE_sam_rm,'o-.','Color',gold_yellow);
plot(T_vec,RMSE_sam_esp,'+--','Color','r');
plot(T_vec,RMSE_UnESPRIT,'s-.','Color','g');
plot(T_vec,RMSE_l1SVD,'*--','Color','m');
% plot(T_vec,RMSE_MLP,'x--','Color','c');
plot(T_vec,RMSE_CNN,'d--','Color','b');
% plot(T_vec, CRLB,'.-','Color','k');
plot(T_vec, CRB_uncr,'.-','Color','k');
hold off;
set(gca, 'YScale', 'log','XScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','UnESPRIT','$\ell_{2,1}$-SVD','CNN','CRLB$_{uncr}$',...
    'interpreter','latex');
title(['DoA-estimation of K=2 sources at ',num2str(SNR), ' dB SNR'], 'interpreter','latex');
ylabel('RMSE [degrees]', 'interpreter','latex');
xlabel('T [snapshots] $\times 100$', 'interpreter','latex');
xticks([1 2 5 10 20 50 100]*100);
xticklabels([1 2 5 10 20 50 100]);
grid on;

% savefig(f,'RMSE_exp2_T100_cnn_res1_fixed_ang_offgrid.fig');