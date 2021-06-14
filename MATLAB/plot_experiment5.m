% Plot the results for Exp. 1B with Delta \theta =2.11 at 0 dB SNR T=200

clear all;
close all;
% Load the CNN results
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\MUSIC_RESULTS\Slide_ang_2coma11sep_K2_0dB_T200_CNN_new_v2_power_mismatch.h5');
gt_ang = h5read(filename, '/GT_angles');
CNN_pred = double(h5read(filename, '/CNN_pred_angles'));

% Load the l1-SVD results 
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1SVD_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11_power_mismatch.h5');
l1_SVD_ang_est = double(h5read(filename2, '/l1_SVD_ang'));
    
sam_ind = 1:length(gt_ang(1,:));

f_cnn(1) = figure(1);
plot(sam_ind, gt_ang(1,:),'Color','r');
hold on;
plot(sam_ind, gt_ang(2,:),'Color','b');
s1 = scatter(sam_ind, CNN_pred(1,:),'Marker','d','MarkerEdgeColor',[1 0 0]);
s2 = scatter(sam_ind, CNN_pred(2,:),'Marker','d','MarkerEdgeColor',[0 0 1]);
hold off;
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
title('CNN', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_cnn(1) = figure(2);
scatter(sam_ind, gt_ang(1,:)-CNN_pred(1,:),'Marker','d','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,gt_ang(2,:)-CNN_pred(2,:),'Marker','d','MarkerEdgeColor',[0 0 1]);
grid on;
title('CNN errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

c1 = gt_ang(1,:)-CNN_pred(1,:);
c2 = gt_ang(2,:)-CNN_pred(2,:);

% Plot l1-SVD results
f_gl1(1) = figure(3);
plot(sam_ind, gt_ang(1,:),'Color','r');
hold on;
plot(sam_ind, gt_ang(2,:),'Color','b');
scatter(sam_ind, l1_SVD_ang_est(1,:),'Marker','*','MarkerEdgeColor',[1 0 0]);
scatter(sam_ind, l1_SVD_ang_est(2,:),'Marker','*','MarkerEdgeColor',[0 0 1]);
hold off;
xlim([0 sam_ind(end)+1]); ylim([-61 61]);
grid on;
title('$\ell_{2,1}$-SVD', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

f_gl1(1) = figure(4);
scatter(sam_ind, gt_ang(1,:)-l1_SVD_ang_est(1,:),'Marker','*','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,gt_ang(2,:)-l1_SVD_ang_est(2,:),'Marker','*','MarkerEdgeColor',[0 0 1]);
grid on;
title('$\ell_{2,1}$-SVD errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(gt_ang - l1_SVD_ang_est).^2;
RMSE_l1_SVD = sqrt(sum(sq_dif(:))/length(sam_ind)/2)

