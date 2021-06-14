% Plot the results for Exp. 1A with Delta \theta =4.7 at -10 dB SNR T=2000

% clear all;
close all;
% Load the MUSIC, R-MUSIC results
File = fullfile(save_path,'Slide_angsep4coma7_K2_min10dB_T2000.mat');
load(File);

% Load the CNN results
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\Slide_angsep4coma7_K2_min10dB_T2000_CNN_new_RQ.h5');
gt_ang = h5read(filename, '/GT_angles');
CNN_pred = double(h5read(filename, '/CNN_pred_angles'));
% Load the l1-SVD results 
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1SVD_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5');
l1_SVD_ang_est = double(h5read(filename2, '/l1_SVD_ang'));

% Load the UnESPRIT results 
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5');
UnESPRIT_ang_est = double(h5read(filename3, '/UnESPRIT_ang'));

sam_ind = 1:length(ang_gt(1,:));

ang_MLP = [[-60.         -54.95325172]
 [-59.20062677 -54.145241  ]
 [-58.43745705 -53.39195214]
 [-56.97693283 -52.14520222]
 [-56.00026085 -51.50710193]
 [-54.94564088 -50.32414197]
 [-54.16057568 -49.68852126]
 [-50.75225578 -50.75225578]
 [-49.31674791 -49.31674791]
 [-48.48219562 -48.48219562]
 [-49.62406448 -45.12758502]
 [-46.63945842 -46.63945842]
 [-47.94209753 -43.09887173]
 [-47.42884815 -42.50206717]
 [-44.64340778 -44.64340778]
 [-44.75118638 -44.75118638]
 [-43.67875365 -39.53402871]
 [-42.17715287 -37.98276638]
 [-41.49863098 -37.6768371 ]
 [-40.23670628 -36.7828932 ]
 [-39.19547923 -35.63989616]
 [-37.         -34.79873864]
 [-38.85322043 -33.68655941]
 [-36.90207452 -33.        ]
 [-36.25631802 -31.14875506]
 [-34.86232267 -30.09949768]
 [-33.81824859 -29.41669643]
 [-33.02989257 -28.15843793]
 [-32.2900907  -27.65980929]
 [-30.73714625 -25.64803219]
 [-30.01072797 -25.34506494]
 [-28.88382504 -23.9296329 ]
 [-27.59953302 -23.02112549]
 [-27.         -22.42724242]
 [-25.85099764 -21.94829869]
 [-22.37939867 -22.37939867]
 [-24.01049552 -19.07425771]
 [-23.         -18.40879061]
 [-21.99513419 -17.09262717]
 [-21.46428608 -16.70603341]
 [-17.25063129 -17.25063129]
 [-18.02813216 -14.60944731]
 [-17.72525816 -13.53566982]
 [-17.4705728  -12.34654891]
 [-16.91069126 -11.99939176]
 [-14.86998832 -10.19932409]
 [-14.14858254  -9.5689672 ]
 [-12.64614033  -8.00017005]
 [-11.71563481  -6.67715496]
 [-11.11065906  -6.30182677]
 [ -9.84020505  -5.26957129]
 [ -6.45058273  -6.45058273]
 [ -8.01336383  -3.48043014]
 [ -7.09130988  -2.28783066]
 [ -6.29821313  -2.3250274 ]
 [ -4.82962312  -0.29279181]
 [ -3.78391358   1.01434433]
 [ -3.           1.66355156]
 [ -2.           2.06935878]
 [ -0.51483533   3.87950648]
 [  1.           4.67659644]
 [  1.5117013    5.89420273]
 [  2.8598589    6.95649509]
 [  2.52391691   7.44788569]
 [  3.62145301   9.48325675]
 [  5.10636215   9.81796517]
 [  4.60290281  10.61819701]
 [  7.16567975  11.8873428 ]
 [  7.86543594  12.51426159]
 [  9.24645184  13.80909683]
 [ 10.21396076  14.80851219]
 [ 11.27040965  15.80998048]
 [ 12.10165598  16.96690854]
 [ 13.17841072  18.13097691]
 [ 14.          18.        ]
 [ 14.939092    19.60433036]
 [ 15.89695943  19.7555497 ]
 [ 17.36149341  21.90239573]
 [ 18.          22.        ]
 [ 20.26195365  24.        ]
 [ 20.52998294  24.28042589]
 [ 21.02288089  25.87181219]
 [ 22.79135476  26.32948514]
 [ 22.70096953  27.08387987]
 [ 23.87689811  28.67094721]
 [ 25.14182335  29.51276484]
 [ 26.          30.59404624]
 [ 26.5734813   30.93755536]
 [ 30.50327904  30.50327904]
 [ 31.71474903  31.71474903]
 [ 29.96282642  34.75096544]
 [ 31.03087047  35.52510966]
 [ 32.008088    37.28751341]
 [ 33.07908135  37.        ]
 [ 35.72760348  35.72760348]
 [ 37.43698603  37.43698603]
 [ 36.62693115  40.07719467]
 [ 37.29327153  40.98516714]
 [ 40.39445886  40.39445886]
 [ 39.1047586   43.09190215]
 [ 40.6892351   44.42594102]
 [ 41.15992375  45.28736091]
 [ 41.68660352  46.79467865]
 [ 45.314652    45.314652  ]
 [ 44.47349269  48.44897928]
 [ 45.46955044  49.27597428]
 [ 45.82532565  51.        ]
 [ 46.75451801  51.69962828]
 [ 50.67106771  50.67106771]
 [ 49.08849586  53.99085085]
 [ 49.82904407  54.18355167]
 [ 50.89193083  56.0343205 ]
 [ 52.40065896  56.32704781]
 [ 52.77983688  57.65350571]
 [ 51.          51.        ]
 [ 55.42036288  55.42036288]]';

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
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
xlim([0 sam_ind(end)+1]); ylim([-30 30]);
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
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
ylim([-2.2 2.2]);
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
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
ylim([-2.2 2.2]);
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
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
ylim([-2.2 2.2]);
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
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
ylim([-2.2 2.2]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('UnESPRIT Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - UnESPRIT_ang_est).^2;
RMSE_UnESPRIT = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

VarNames = {'RMSE'};
Tab = table(VarNames, RMSE_MUSIC, RMSE_RMUSIC, RMSE_CNN, RMSE_l1_SVD, RMSE_MLP, RMSE_ESPRIT, RMSE_UnESPRIT)

