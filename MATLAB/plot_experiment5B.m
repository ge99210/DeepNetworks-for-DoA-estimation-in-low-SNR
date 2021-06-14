% Plot the results for Exp. 1A with Delta \theta =4.7 at -10 dB SNR T=2000

% clear all;
close all;
% Load the MUSIC, R-MUSIC results
File = fullfile(save_path,'Slide_angsep4_K2_min10dB_T1000_power_mismatch.mat');
load(File);

% Load the CNN results
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\Slide_angsep4_K2_min10dB_T1000_CNN_new_power_mismatch_new_lr_RoP_0_7.h5');
gt_ang = h5read(filename, '/GT_angles');
CNN_pred = double(h5read(filename, '/CNN_pred_angles'));
% Load the l1-SVD results 
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1SVD_16ULA_K2_min10dBSNR_T1000_3D_slideang_offgrid_sep4_power_mismatch.h5');
l1_SVD_ang_est = double(h5read(filename2, '/l1_SVD_ang'));
% Load the UnESPRIT results 
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T1000_3D_slideang_offgrid_sep4_power_mismatch.h5');
UnESPRIT_ang_est = double(h5read(filename3, '/UnESPRIT_ang'));

sam_ind = 1:length(ang_gt(1,:));

ang_MLP = [[-56.66383593 -56.66383593]
 [-57.93171953 -53.49580901]
 [-57.07420141 -52.68062486]
 [-53.88689233 -53.88689233]
 [-52.57404479 -52.57404479]
 [-53.23725259 -50.52473751]
 [-52.86335103 -48.18021838]
 [-49.69738952 -49.69738952]
 [-50.37131678 -47.12392936]
 [-50.04449757 -45.49573567]
 [-46.90820301 -46.90820301]
 [-47.90561136 -43.21465932]
 [-47.12561253 -43.12888348]
 [-44.21555694 -44.21555694]
 [-44.70405824 -44.70405824]
 [-43.0499076  -39.61626256]
 [-42.         -38.98232979]
 [-40.96252196 -37.97553695]
 [-37.04676118 -37.04676118]
 [-38.84753337 -36.        ]
 [-37.98689514 -34.82818639]
 [-40.26163105 -35.14007074]
 [-36.32883379 -33.        ]
 [-35.55244104 -31.61489734]
 [-37.         -32.68925993]
 [-31.68979035 -31.68979035]
 [-32.69389119 -28.741085  ]
 [-32.34971053 -28.08181988]
 [-30.65725666 -26.46404577]
 [-27.54885333 -27.54885333]
 [-28.5071324  -24.81220969]
 [-27.31369887 -23.97205824]
 [-26.2907655  -22.69248745]
 [-25.74893796 -21.82427502]
 [-23.1580674  -23.1580674 ]
 [-23.03451701 -20.04270968]
 [-20.8490462  -20.8490462 ]
 [-19.54980999 -19.54980999]
 [-18.04769177 -18.04769177]
 [-20.02009021 -16.52529796]
 [-16.07560847 -16.07560847]
 [-16.34616026 -13.79262912]
 [-17.14243016 -12.92816005]
 [-18.75001533 -13.61578465]
 [-14.40803182 -10.56376367]
 [-11.9289007  -11.9289007 ]
 [-12.35224693  -8.82393554]
 [-11.74494976  -8.        ]
 [-10.89767215  -6.64437959]
 [ -9.5538088   -5.85676295]
 [ -6.84168297  -6.84168297]
 [ -7.90292959  -3.66051514]
 [ -6.49994584  -2.5819319 ]
 [ -5.21315286  -1.43009301]
 [ -4.74698294  -0.95133458]
 [ -3.44732459  -0.08695468]
 [ -2.73029565   1.00980478]
 [ -1.6506427    2.        ]
 [  1.           3.02453208]
 [  3.09458111   3.09458111]
 [  1.           4.79869021]
 [  4.24421345   4.24421345]
 [  5.54326813   5.54326813]
 [  4.6636275    7.82779548]
 [  5.49638816   8.97089854]
 [  6.56508688  10.26279062]
 [  7.5877268   11.48915022]
 [ 10.29672829  10.29672829]
 [ 11.32982894  11.32982894]
 [ 10.16135119  14.21313755]
 [ 13.66713695  13.66713695]
 [ 12.87085432  16.73897964]
 [ 13.42145284  17.78395488]
 [ 15.54530171  15.54530171]
 [ 15.82367776  18.64970788]
 [ 16.69619432  16.69619432]
 [ 18.          20.52374057]
 [ 20.3850445   20.3850445 ]
 [ 18.98123551  23.04122923]
 [ 21.44094522  24.        ]
 [ 21.56651296  25.        ]
 [ 23.02926508  26.16227127]
 [ 25.33036547  25.33036547]
 [ 24.52867836  28.20260747]
 [ 26.          29.1095411 ]
 [ 26.          29.60896106]
 [ 27.41881925  31.15064963]
 [ 28.80330802  31.78335989]
 [ 29.37599141  33.15633394]
 [ 29.79644317  34.52393256]
 [ 31.47140296  36.10455531]
 [ 32.98768644  35.84391571]
 [ 33.05211947  37.58999614]
 [ 34.39492319  38.02464228]
 [ 35.          39.10312034]
 [ 38.53395289  38.53395289]
 [ 37.87131461  40.99859296]
 [ 40.33177988  40.33177988]
 [ 41.66099946  41.66099946]
 [ 42.5788851   42.5788851 ]
 [ 43.38104639  43.38104639]
 [ 44.26360862  44.26360862]
 [ 44.97611453  44.97611453]
 [ 43.53312846  48.2118234 ]
 [ 47.41948809  47.41948809]
 [ 45.98699877  50.46108362]
 [ 47.3885935   51.        ]
 [ 48.90437119  53.        ]
 [ 51.34865265  51.34865265]
 [ 50.90826178  54.13713159]
 [ 52.8977947   52.8977947 ]
 [ 53.9805352   53.9805352 ]
 [ 52.86535347  57.21526999]
 [ 46.99137809  46.99137809]
 [ 55.43316124  55.43316124]
 [ 57.          57.        ]]';

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
ylim([-10 10]);
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
ylim([-10 10]);
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
ylim([-10 10]);
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
ylim([-10 10]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('UnESPRIT Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - UnESPRIT_ang_est).^2;
RMSE_UnESPRIT = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

VarNames = {'RMSE'};
Tab = table(VarNames, RMSE_MUSIC, RMSE_RMUSIC, RMSE_CNN, RMSE_l1_SVD, RMSE_MLP, RMSE_ESPRIT, RMSE_UnESPRIT)

