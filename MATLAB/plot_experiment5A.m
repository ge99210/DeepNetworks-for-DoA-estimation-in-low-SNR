% Plot the results for Exp. 1A with Delta \theta =4.7 at -10 dB SNR T=2000

% clear all;
close all;
% Load the MUSIC, R-MUSIC results
File = fullfile(save_path,'Slide_angsep2coma11_K2_0dB_T200_power_mismatch.mat');
load(File);

% Load the CNN results
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\Slide_ang_2coma11sep_K2_0dB_T200_CNN_new_power_mismatch_new.h5');
gt_ang = h5read(filename, '/GT_angles');
CNN_pred = double(h5read(filename, '/CNN_pred_angles'));
% Load the l1-SVD results 
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1SVD_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11_power_mismatch.h5');
l1_SVD_ang_est = double(h5read(filename2, '/l1_SVD_ang'));
% Load the UnESPRIT results 
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11_power_mismatch.h5');
UnESPRIT_ang_est = double(h5read(filename3, '/UnESPRIT_ang'));

sam_ind = 1:length(ang_gt(1,:));

ang_MLP = [[-5.82871103e+01 -5.82871103e+01]
 [-5.72440345e+01 -5.72440345e+01]
 [-5.81588167e+01 -5.47137254e+01]
 [-5.60993920e+01 -5.40000000e+01]
 [-5.50353550e+01 -5.29074564e+01]
 [-5.40023698e+01 -5.18319143e+01]
 [-5.24492640e+01 -5.24492640e+01]
 [-5.12085140e+01 -5.12085140e+01]
 [-5.01805577e+01 -5.01805577e+01]
 [-5.00000000e+01 -4.78069945e+01]
 [-4.79792796e+01 -4.79792796e+01]
 [-4.70390093e+01 -4.70390093e+01]
 [-4.61576331e+01 -4.61576331e+01]
 [-4.51867779e+01 -4.51867779e+01]
 [-4.42611366e+01 -4.42611366e+01]
 [-4.50000000e+01 -4.18032573e+01]
 [-4.24488379e+01 -4.24488379e+01]
 [-4.10968117e+01 -4.10968117e+01]
 [-4.10030556e+01 -3.90000000e+01]
 [-3.80000000e+01 -3.80000000e+01]
 [-3.95459585e+01 -3.69653358e+01]
 [-3.71249388e+01 -3.71249388e+01]
 [-3.59083224e+01 -3.59083224e+01]
 [-3.64897022e+01 -3.40000000e+01]
 [-3.42920189e+01 -3.42920189e+01]
 [-3.41169367e+01 -3.20000000e+01]
 [-3.30310719e+01 -3.09675751e+01]
 [-3.10763928e+01 -3.10763928e+01]
 [-2.99940118e+01 -2.99940118e+01]
 [-3.00000000e+01 -2.78719043e+01]
 [-2.83235709e+01 -2.83235709e+01]
 [-2.74139172e+01 -2.74139172e+01]
 [-2.73953963e+01 -2.46537319e+01]
 [-2.63850254e+01 -2.37803586e+01]
 [-2.42781940e+01 -2.42781940e+01]
 [-2.33192831e+01 -2.33192831e+01]
 [-2.23101626e+01 -2.23101626e+01]
 [-2.09701522e+01 -2.09701522e+01]
 [-2.10114265e+01 -1.90000000e+01]
 [-1.84304978e+01 -1.84304978e+01]
 [-1.90000000e+01 -1.69891793e+01]
 [-1.69127585e+01 -1.69127585e+01]
 [-1.90000000e+01 -1.56798194e+01]
 [-1.60000000e+01 -1.40000000e+01]
 [-1.40028591e+01 -1.40028591e+01]
 [-1.28125837e+01 -1.28125837e+01]
 [-1.20774778e+01 -1.20774778e+01]
 [-1.20000000e+01 -1.00000000e+01]
 [-1.11191880e+01 -8.99478338e+00]
 [-9.56128574e+00 -9.56128574e+00]
 [-9.00000000e+00 -7.00000000e+00]
 [-8.00411269e+00 -5.99425671e+00]
 [-6.19413643e+00 -6.19413643e+00]
 [-6.00000000e+00 -4.00000000e+00]
 [-5.04164223e+00 -3.00000000e+00]
 [-3.54430813e+00 -3.54430813e+00]
 [-2.63303936e+00 -2.63303936e+00]
 [-7.95485131e-01 -7.95485131e-01]
 [-1.00000000e+00  1.07065543e+00]
 [-5.07220793e-02  2.00000000e+00]
 [ 2.13988487e+00  2.13988487e+00]
 [ 0.00000000e+00  4.00000000e+00]
 [ 3.64128086e+00  3.64128086e+00]
 [ 4.39515846e+00  4.39515846e+00]
 [ 5.65637341e+00  5.65637341e+00]
 [ 5.96460254e+00  8.00000000e+00]
 [ 7.00000000e+00  9.00000000e+00]
 [ 8.70646976e+00  8.70646976e+00]
 [ 8.83571012e+00  1.10000000e+01]
 [ 1.07600109e+01  1.07600109e+01]
 [ 1.09492487e+01  1.33528551e+01]
 [ 1.19834467e+01  1.40000000e+01]
 [ 1.30000000e+01  1.50000000e+01]
 [ 1.50423526e+01  1.50423526e+01]
 [ 1.50000000e+01  1.70415324e+01]
 [ 1.59473875e+01  1.80000000e+01]
 [ 1.70000000e+01  1.90000000e+01]
 [ 1.82967001e+01  1.82967001e+01]
 [ 1.90000000e+01  2.10795026e+01]
 [ 1.98833374e+01  2.21255691e+01]
 [ 2.24455896e+01  2.24455896e+01]
 [ 2.14780710e+01  2.40000000e+01]
 [ 2.35668779e+01  2.35668779e+01]
 [ 2.47486663e+01  2.47486663e+01]
 [ 2.56866863e+01  2.56866863e+01]
 [ 2.68419733e+01  2.68419733e+01]
 [ 2.68993623e+01  2.90000000e+01]
 [ 2.79780583e+01  3.00000000e+01]
 [ 2.90000000e+01  3.10000000e+01]
 [ 3.08075827e+01  3.08075827e+01]
 [ 3.18379087e+01  3.18379087e+01]
 [ 3.29411539e+01  3.29411539e+01]
 [ 3.40962015e+01  3.40962015e+01]
 [ 3.51488092e+01  3.51488092e+01]
 [ 3.56168278e+01  3.56168278e+01]
 [ 3.59298703e+01  3.81284144e+01]
 [ 3.77011525e+01  3.77011525e+01]
 [ 3.89250695e+01  3.89250695e+01]
 [ 3.88769291e+01  4.10000000e+01]
 [ 4.20000000e+01  4.20000000e+01]
 [ 4.27805369e+01  4.27805369e+01]
 [ 4.13180983e+01  4.40000000e+01]
 [ 4.37842850e+01  4.37842850e+01]
 [ 4.49947659e+01  4.49947659e+01]
 [ 4.62784143e+01  4.62784143e+01]
 [ 4.69955055e+01  4.69955055e+01]
 [ 4.70000000e+01  4.90000000e+01]
 [ 4.81484393e+01  4.81484393e+01]
 [ 4.95365969e+01  4.95365969e+01]
 [ 5.04834273e+01  5.04834273e+01]
 [ 5.15317439e+01  5.15317439e+01]
 [ 5.27301896e+01  5.27301896e+01]
 [ 5.35873908e+01  5.35873908e+01]
 [ 5.39312330e+01  5.60000000e+01]
 [ 5.60658887e+01  5.60658887e+01]
 [ 5.70537604e+01  5.70537604e+01]
 [ 5.15902057e+01  5.15902057e+01]
 [ 5.70000000e+01  5.70000000e+01]]';

count = 0;
for ii=1:size(sam_ind,2)
    if ang_MLP(1,ii)==ang_MLP(2,ii)
        ang_MLP(2,ii) = NaN;
        count = count + 1;
    end
end
% Plot the MUSIC results
f_m(1)= figure(1); 
%subplot(2,1,1);
plot(sam_ind, ang_gt(1,:),'Color','r');
hold on;
plot(sam_ind, ang_gt(2,:),'Color','b');
scatter(sam_ind, ang_sam(1,:),'Marker','^','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, ang_sam(2,:),'Marker','^','MarkerEdgeColor',[0 0 1]);
hold off;
% xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
legend('$\theta_1$','$\theta_2$','$\hat{\theta}_1$','$\hat{\theta}_2$','interpreter','latex');
title('MUSIC', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
%xlabel('Sample index', 'interpreter','latex');

f_m(2) = figure(2);
%subplot(2,1,2);
scatter(sam_ind, ang_gt(1,:)-ang_sam(1,:),'Marker','^','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind, ang_gt(2,:)-ang_sam(2,:),'Marker','^','MarkerEdgeColor',[0 0 1]);
hold off;
grid on;
% xlim([0 sam_ind(end)+1]); ylim([-30 30]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('MUSIC Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - ang_sam).^2;
RMSE_MUSIC = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot the R-MUSIC results
f_rm(1) = figure(3);
plot(sam_ind, ang_gt(1,:),'Color','r');
hold on;
plot(sam_ind, ang_gt(2,:),'Color','b');
scatter(sam_ind, ang_sam_rm(1,:),'Marker','o','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, ang_sam_rm(2,:),'Marker','o','MarkerEdgeColor',[0 0 1]);
hold off;
% xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
legend('$\theta_1$','$\theta_2$','$\hat{\theta}_1$','$\hat{\theta}_2$','interpreter','latex');
title('R-MUSIC', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_rm(2) = figure(4);
scatter(sam_ind, ang_gt(1,:)-ang_sam_rm(1,:),'Marker','o','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,ang_gt(2,:)-ang_sam_rm(2,:),'Marker','o','MarkerEdgeColor',[0 0 1]);
grid on;
% ylim([-2.2 2.2]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('R-MUSIC errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - ang_sam_rm).^2;
RMSE_RMUSIC = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot the proposed CNN results
f_cnn(1) = figure(5);
plot(sam_ind, gt_ang(1,:),'Color','r');
hold on;
plot(sam_ind, gt_ang(2,:),'Color','b');
scatter(sam_ind, CNN_pred(1,:),'Marker','d','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, CNN_pred(2,:),'Marker','d','MarkerEdgeColor',[0 0 1]);
hold off;
% xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
legend('$\theta_1$','$\theta_2$','$\hat{\theta}_1$','$\hat{\theta}_2$','interpreter','latex');
title('CNN', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_cnn(2) = figure(6);
scatter(sam_ind, gt_ang(1,:)-CNN_pred(1,:),'Marker','d','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,gt_ang(2,:)-CNN_pred(2,:),'Marker','d','MarkerEdgeColor',[0 0 1]);
grid on;
ylim([-13 13]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('CNN errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - CNN_pred).^2;
RMSE_CNN = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot l1-SVD results
f_gl1(1) = figure(7);
plot(sam_ind, gt_ang(1,:),'Color','r');
hold on;
plot(sam_ind, gt_ang(2,:),'Color','b');
scatter(sam_ind, l1_SVD_ang_est(1,:),'Marker','*','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, l1_SVD_ang_est(2,:),'Marker','*','MarkerEdgeColor',[0 0 1]);
hold off;
% xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
legend('$\theta_1$','$\theta_2$','$\hat{\theta}_1$','$\hat{\theta}_2$','interpreter','latex');
title('$\ell_{2,1}$-SVD', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_gl1(2) = figure(8);
scatter(sam_ind, gt_ang(1,:)-l1_SVD_ang_est(1,:),'Marker','*','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,gt_ang(2,:)-l1_SVD_ang_est(2,:),'Marker','*','MarkerEdgeColor',[0 0 1]);
grid on;
ylim([-13 13]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('$\ell_{2,1}$-SVD errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - l1_SVD_ang_est).^2;
RMSE_l1_SVD = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot the MLP results
f_mlp(1) = figure(9);
plot(sam_ind, gt_ang(1,:),'Color','r');
hold on;
plot(sam_ind, gt_ang(2,:),'Color','b');
scatter(sam_ind, ang_MLP(1,:),'Marker','x','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, ang_MLP(2,:),'Marker','x','MarkerEdgeColor',[0 0 1]);
hold off;
% xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
legend('$\theta_1$','$\theta_2$','$\hat{\theta}_1$','$\hat{\theta}_2$','interpreter','latex');
title('MLP', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_mlp(2) = figure(10);
scatter(sam_ind, gt_ang(1,:)-ang_MLP(1,:),'Marker','x','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,gt_ang(2,:)-ang_MLP(2,:),'Marker','x','MarkerEdgeColor',[0 0 1]);
grid on;
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('MLP errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - ang_MLP).^2;
RMSE_MLP = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot the ESPRIT results
f_esp(1) = figure(11);
plot(sam_ind, ang_gt(1,:),'Color','r');
hold on;
plot(sam_ind, ang_gt(2,:),'Color','b');
scatter(sam_ind, ang_sam_esp(1,:),'Marker','+','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, ang_sam_esp(2,:),'Marker','+','MarkerEdgeColor',[0 0 1]);
hold off;
% xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
legend('$\theta_1$','$\theta_2$','$\hat{\theta}_1$','$\hat{\theta}_2$','interpreter','latex');
title('ESPRIT', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_esp(2) = figure(12);
scatter(sam_ind, ang_gt(1,:)-ang_sam_esp(1,:),'Marker','+','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,ang_gt(2,:)-ang_sam_esp(2,:),'Marker','+','MarkerEdgeColor',[0 0 1]);
grid on;
ylim([-13 13]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('ESPRIT Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - ang_sam_esp).^2;
RMSE_ESPRIT = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot UnESPRIT results
f_unesp(1) = figure(13);
plot(sam_ind, gt_ang(1,:),'Color','r');
hold on;
plot(sam_ind, gt_ang(2,:),'Color','b');
scatter(sam_ind, UnESPRIT_ang_est(1,:),'Marker','s','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, UnESPRIT_ang_est(2,:),'Marker','s','MarkerEdgeColor',[0 0 1]);
hold off;
% xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
legend('$\theta_1$','$\theta_2$','$\hat{\theta}_1$','$\hat{\theta}_2$','interpreter','latex');
title('UnESPRIT', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_unesp(2) = figure(14);
scatter(sam_ind, gt_ang(1,:)-UnESPRIT_ang_est(1,:),'Marker','s','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,gt_ang(2,:)-UnESPRIT_ang_est(2,:),'Marker','s','MarkerEdgeColor',[0 0 1]);
grid on;
ylim([-13 13]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('UnESPRIT Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - UnESPRIT_ang_est).^2;
RMSE_UnESPRIT = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

VarNames = {'RMSE'};
Tab = table(VarNames, RMSE_MUSIC, RMSE_RMUSIC, RMSE_CNN, RMSE_l1_SVD, RMSE_MLP, RMSE_ESPRIT, RMSE_UnESPRIT)

