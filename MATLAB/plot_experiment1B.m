% Plot the results for Exp. 1B with Delta \theta =2.11 at 0 dB SNR T=200

close all;

% Load the MUSIC, R-MUSIC results
File = fullfile(save_path,'Slide_angsep2coma11_K2_0dB_T200.mat');
load(File);

% Load the CNN results
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\Slide_ang_2coma11sep_K2_0dB_T200_CNN_RQ.h5');
gt_ang = h5read(filename, '/GT_angles');
CNN_pred = double(h5read(filename, '/CNN_pred_angles'));

% Load the l1-SVD results 
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1SVD_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11.h5');
l1_SVD_ang_est = double(h5read(filename2, '/l1_SVD_ang'));
    
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11.h5');
UnESPRIT_ang_est = double(h5read(filename3, '/UnESPRIT_ang'));

sam_ind = 1:length(ang_gt(1,:));

ang_MLP = [[-58.58102006 -58.58102006]
 [-57.61889195 -57.61889195]
 [-58.19135518 -54.9332847 ]
 [-55.57817099 -55.57817099]
 [-54.16059591 -54.16059591]
 [-52.98113591 -52.98113591]
 [-52.45154147 -52.45154147]
 [-51.71131289 -51.71131289]
 [-50.69628834 -50.69628834]
 [-50.         -48.        ]
 [-48.86888944 -48.86888944]
 [-47.43191255 -47.43191255]
 [-46.61173559 -46.61173559]
 [-47.1106154  -43.86046424]
 [-46.         -43.11792146]
 [-45.0961141  -42.02897901]
 [-44.         -41.63408055]
 [-42.79113952 -42.79113952]
 [-40.56349292 -40.56349292]
 [-38.20852076 -38.20852076]
 [-36.97515013 -36.97515013]
 [-40.22553301 -36.        ]
 [-36.40675975 -36.40675975]
 [-35.63105605 -35.63105605]
 [-34.33382946 -34.33382946]
 [-33.53758696 -33.53758696]
 [-32.40128099 -32.40128099]
 [-31.12022665 -31.12022665]
 [-32.18764396 -28.00154305]
 [-29.13135302 -29.13135302]
 [-28.45969951 -28.45969951]
 [-27.68763118 -27.68763118]
 [-26.14548707 -26.14548707]
 [-26.66898992 -23.46531302]
 [-24.59492882 -24.59492882]
 [-23.67352982 -23.67352982]
 [-22.90366696 -22.90366696]
 [-21.66044068 -21.66044068]
 [-20.11869848 -20.11869848]
 [-20.15515174 -17.93250735]
 [-17.86481448 -17.86481448]
 [-17.43792935 -17.43792935]
 [-19.12055683 -15.7092554 ]
 [-15.43008091 -15.43008091]
 [-14.59758195 -14.59758195]
 [-13.10709348 -13.10709348]
 [-12.28534529 -12.28534529]
 [-10.93460963 -10.93460963]
 [-10.36298512 -10.36298512]
 [ -9.84996525  -9.84996525]
 [ -8.31301623  -8.31301623]
 [ -7.63883178  -7.63883178]
 [ -6.52650228  -6.52650228]
 [ -4.88421824  -4.88421824]
 [ -4.58288538  -4.58288538]
 [ -3.86404152  -3.86404152]
 [ -3.02936083  -3.02936083]
 [ -1.92266242  -1.92266242]
 [ -0.35529619  -0.35529619]
 [  0.61346991   0.61346991]
 [  1.82207768   1.82207768]
 [  2.25822512   2.25822512]
 [  2.9659342    2.9659342 ]
 [  2.83869626   5.16075993]
 [  5.27132226   5.27132226]
 [  6.62322701   6.62322701]
 [  7.57031548   7.57031548]
 [  8.33383421   8.33383421]
 [  9.52618255   9.52618255]
 [ 10.57415175  10.57415175]
 [ 11.55732867  11.55732867]
 [ 12.65932739  12.65932739]
 [ 13.55592154  13.55592154]
 [ 14.87624517  14.87624517]
 [ 15.38759913  15.38759913]
 [ 16.23465662  16.23465662]
 [ 17.16412041  17.16412041]
 [ 18.01867796  18.01867796]
 [ 19.80236106  19.80236106]
 [ 20.90744346  20.90744346]
 [ 22.24825617  22.24825617]
 [ 22.60875024  22.60875024]
 [ 23.27341151  23.27341151]
 [ 24.22615544  24.22615544]
 [ 25.14407527  25.14407527]
 [ 26.41995753  26.41995753]
 [ 27.53827824  27.53827824]
 [ 27.7902021   30.        ]
 [ 30.35031397  30.35031397]
 [ 29.95545127  29.95545127]
 [ 31.60182599  31.60182599]
 [ 32.06828949  32.06828949]
 [ 32.          34.8760882 ]
 [ 34.71229432  34.71229432]
 [ 35.58081416  35.58081416]
 [ 36.36880139  36.36880139]
 [ 37.4873462   37.4873462 ]
 [ 38.17588795  38.17588795]
 [ 39.5320504   39.5320504 ]
 [ 39.02946058  42.        ]
 [ 42.86866295  42.86866295]
 [ 42.56354069  42.56354069]
 [ 43.57557918  43.57557918]
 [ 44.2671976   44.2671976 ]
 [ 46.26550916  46.26550916]
 [ 46.5437715   46.5437715 ]
 [ 47.27503439  47.27503439]
 [ 48.10630679  48.10630679]
 [ 49.41960671  49.41960671]
 [ 50.37505687  50.37505687]
 [ 51.28303179  51.28303179]
 [ 45.          45.        ]
 [ 53.46994853  53.46994853]
 [ 54.34671526  54.34671526]
 [ 55.61692839  55.61692839]
 [  8.43923097   8.43923097]
 [ 51.6601211   51.6601211 ]
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
xlabel('Sample index', 'interpreter','latex');

f_m(2) = figure(2);
scatter(sam_ind, ang_gt(1,:)-ang_sam(1,:),'Marker','^','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind, ang_gt(2,:)-ang_sam(2,:),'Marker','^','MarkerEdgeColor',[0 0 1]);
hold off;
grid on;
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('MUSIC errors', 'interpreter','latex');
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
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('R-MUSIC errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - ang_sam_rm).^2;
RMSE_RMUSIC = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% CNN 
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
ylim([-11 11]);
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
%xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
ylim([-11 8]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('$\ell_{2,1}$-SVD errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - l1_SVD_ang_est).^2;
RMSE_l1_SVD = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot the proposed MLP results
f_mlp(1) = figure(9);
plot(sam_ind, gt_ang(1,:),'Color','r');
hold on;
plot(sam_ind, gt_ang(2,:),'Color','b');
scatter(sam_ind, ang_MLP(1,:),'Marker','x','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, ang_MLP(2,:),'Marker','x','MarkerEdgeColor',[0 0 1]);
hold off;
%xlim([0 sam_ind(end)+1]); ylim([-61 61]);
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
xlim([0 sam_ind(end)+1]); ylim([-70 70]);
grid on;
%axis('tight');
legend('$\theta_1$','$\theta_2$','$\hat{\theta}_1$','$\hat{\theta}_2$','interpreter','latex');
title('ESPRIT', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_esp(2) = figure(12);
scatter(sam_ind, ang_gt(1,:)-ang_sam_esp(1,:),'Marker','+','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,ang_gt(2,:)-ang_sam_esp(2,:),'Marker','+','MarkerEdgeColor',[0 0 1]);
grid on;
ylim([-11 8]);
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
xlim([0 sam_ind(end)+1]); ylim([-65 65]);
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
ylim([-11 8]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('UnESPRIT Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - UnESPRIT_ang_est).^2;
RMSE_UnESPRIT = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

VarNames = {'RMSE'};
Tab = table(VarNames, RMSE_MUSIC, RMSE_RMUSIC, RMSE_CNN, RMSE_l1_SVD, RMSE_MLP, RMSE_ESPRIT, RMSE_UnESPRIT)


