% Plot the results for Exp. 5B with Delta \theta =4 at -10 dB SNR T=1000

clear all;
close all;
% Load the CNN results
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\Slide_ang_sep10_K1to3_0dB_T1000_CNN_RQ.h5');
gt_ang1 = h5read(filename, '/GT_angles1');
CNN_pred1 = double(h5read(filename, '/CNN_pred_angles1'));
sam_ind1 = 1:length(gt_ang1);

gt_ang2 = h5read(filename, '/GT_angles2');
CNN_pred2 = double(h5read(filename, '/CNN_pred_angles2'));
sam_ind2 = 1:length(gt_ang2);

gt_ang3 = h5read(filename, '/GT_angles3');
CNN_pred3 = double(h5read(filename, '/CNN_pred_angles3'));
sam_ind3 = 1:length(gt_ang3);

% Manually correcting these values for display purposes only
% K=1
FN_ind1 = gt_ang1(27);

% K=2
CNN_pred2(:,109) = flip(CNN_pred2(:,109));
FN_ind2 = [gt_ang2(2,105) gt_ang2(1,109)];

% K=3
CNN_pred3(3,16) = CNN_pred3(2,16);
CNN_pred3(2,16) = NaN;
FN_ind3 = gt_ang3(2,16);

figure(1);
plot(sam_ind1, gt_ang1,'Color','#0d47a1');
hold on;
scatter(sam_ind1, CNN_pred1(1,:),'Marker','d','MarkerEdgeColor','#9fa8da');
scatter(27, FN_ind1,'Marker','+','MarkerEdgeColor','#9c27b0');
scatter(36, CNN_pred1(2,36),'Marker','d','MarkerEdgeColor','k');
hold off;
xlim([0 sam_ind1(end)+1]); 
ylim([-61 61]);
legend('$\theta_1$', '$\hat{\theta}_1$','FN','FP',...
    'interpreter','latex','location','northwest')
grid on;
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

figure(2);
plot(sam_ind2, gt_ang2(1,:),'Color','#0d47a1');
hold on;
plot(sam_ind2, gt_ang2(2,:),'Color','#c62828');
scatter(sam_ind2, CNN_pred2(1,:),'Marker','d','MarkerEdgeColor','#9fa8da');
scatter(sam_ind2, CNN_pred2(2,:),'Marker','d','MarkerEdgeColor','#ef9a9a');
scatter(105, FN_ind2(1),'Marker','+','MarkerEdgeColor','#9c27b0');
scatter(109, FN_ind2(2),'Marker','+','MarkerEdgeColor','#9c27b0');
hold off;
xlim([0 sam_ind2(end)+1]); 
ylim([-61 61]);
legend('$\theta_1$', '$\theta_2$',...
    '$\hat{\theta}_1$', '$\hat{\theta}_2$','FN',...
    'interpreter','latex','location','northwest');
grid on;
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

figure(3);
plot(sam_ind3, gt_ang3(1,:),'Color','#0d47a1');
hold on;
plot(sam_ind3, gt_ang3(2,:),'Color','#c62828');
plot(sam_ind3, gt_ang3(3,:),'Color','#689f38');
scatter(sam_ind3, CNN_pred3(1,:),'Marker','d','MarkerEdgeColor','#9fa8da');
scatter(sam_ind3, CNN_pred3(2,:),'Marker','d','MarkerEdgeColor','#ef9a9a');
scatter(sam_ind3, CNN_pred3(3,:),'Marker','d','MarkerEdgeColor','#c5e1a5');
scatter(16, FN_ind3,'Marker','+','MarkerEdgeColor','#9c27b0');
hold off;
xlim([0 sam_ind3(end)+1]); 
ylim([-61 61]);
legend('$\theta_1$', '$\theta_2$','$\theta_3$',...
    '$\hat{\theta}_1$', '$\hat{\theta}_2$','$\hat{\theta}_3$','FN',...
    'interpreter','latex','location','northwest');
grid on;
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');
