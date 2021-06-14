% Plot the results for Exp. 1A with Delta \theta =4.7 at -10 dB SNR T=2000

% clear all;
close all;
% Load the MUSIC, R-MUSIC results
File = fullfile(save_path,'Slide_angsep4coma7_K2_min10dB_T2000_90deg.mat');
load(File);

% Load the CNN results
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation',...
        'DoA_Estimation_underdetermined\ComparisonRESULTS\Slide_angsep4coma7_K2_min10dB_T2000_CNN_new_RQ_90deg_vf6c.h5');
ang_gt = h5read(filename, '/GT_angles');
CNN_pred = double(h5read(filename, '/CNN_pred_angles'));

% Load the l1-SVD results 
filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_l1SVD_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7_90deg.h5');
l1_SVD_ang_est = double(h5read(filename2, '/l1_SVD_ang'));

% Load the UnESPRIT results 
filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7_90deg.h5');
UnESPRIT_ang_est = double(h5read(filename3, '/UnESPRIT_ang'));

sam_ind = 1:length(ang_gt(1,:));

ang_MLP = [[-85.45373277 -85.45373277]
 [-85.24155328 -85.24155328]
 [-84.78905905 -84.78905905]
 [-81.29537787 -81.29537787]
 [-84.0543271  -84.0543271 ]
 [-82.05464057 -82.05464057]
 [-80.68521153 -80.68521153]
 [-79.57336084 -79.57336084]
 [-78.94015489 -78.94015489]
 [-78.67703493 -78.67703493]
 [-82.8542354  -75.571186  ]
 [-81.676099   -74.82832297]
 [-80.43141725 -72.86055832]
 [-74.44175858 -74.44175858]
 [-72.74328181 -72.74328181]
 [-71.74148404 -71.74148404]
 [-77.17940924 -69.14153742]
 [-71.71264562 -71.71264562]
 [-69.40690856 -69.40690856]
 [-68.82295058 -68.82295058]
 [-68.21899692 -68.21899692]
 [-67.30849054 -67.30849054]
 [-67.53040056 -67.53040056]
 [-65.66238372 -65.66238372]
 [-65.37594494 -65.37594494]
 [-64.91964003 -64.91964003]
 [-61.48205505 -61.48205505]
 [-62.7558642  -58.98157685]
 [-61.63462797 -57.51640507]
 [-61.41815096 -56.97391301]
 [-56.7864631  -56.7864631 ]
 [-59.22904105 -54.89570623]
 [-57.57420554 -52.79415261]
 [-57.23562364 -52.13347926]
 [-53.48059866 -53.48059866]
 [-52.61486824 -52.61486824]
 [-54.67725025 -49.27691065]
 [-53.07113285 -48.46315012]
 [-51.98941081 -47.11395421]
 [-50.96562032 -46.31265565]
 [-49.7909687  -44.94346933]
 [-48.77300154 -44.57883989]
 [-46.49308319 -43.        ]
 [-46.8187371  -42.28720589]
 [-45.75184233 -41.33004115]
 [-44.85992754 -40.19738922]
 [-41.36086802 -41.36086802]
 [-40.58735813 -40.58735813]
 [-42.85979311 -37.73961259]
 [-41.10255021 -36.40159426]
 [-40.08383749 -36.44905617]
 [-38.89950489 -34.25282251]
 [-38.23029554 -33.76916454]
 [-37.         -32.45837337]
 [-36.03326134 -32.30580226]
 [-34.60905683 -30.        ]
 [-34.         -29.86445142]
 [-32.99705777 -28.58577765]
 [-29.99119891 -29.99119891]
 [-30.3155816  -26.09534042]
 [-25.97216559 -25.97216559]
 [-29.84604065 -23.94427677]
 [-26.29077318 -22.85445295]
 [-25.35228376 -25.35228376]
 [-27.90705333 -22.56800314]
 [-22.06483735 -22.06483735]
 [-22.60932281 -22.60932281]
 [-23.56068262 -18.54766934]
 [-19.6840267  -19.6840267 ]
 [-21.00848999 -16.43207587]
 [-20.85408864 -15.8121992 ]
 [-19.1940716  -15.65307217]
 [-17.74735081 -14.58916992]
 [-14.97489097 -14.97489097]
 [-14.0872324  -14.0872324 ]
 [-12.71626881 -12.71626881]
 [-13.08930676  -8.39201477]
 [-10.26662016 -10.26662016]
 [ -9.9658805   -9.9658805 ]
 [-11.17991763  -6.82877491]
 [-12.          -7.48638529]
 [ -8.55682902  -3.31424609]
 [ -5.05273202  -5.05273202]
 [ -7.          -2.68718083]
 [ -4.30258783  -4.30258783]
 [ -4.66007643   0.        ]
 [ -3.57556658   1.04091151]
 [ -3.0367755    2.00984587]
 [ -2.0779625    2.84691492]
 [ -1.           4.00274163]
 [  0.26340908   4.1176782 ]
 [ -0.15221255   4.47150816]
 [  1.75116145   6.55616302]
 [  1.71356647   7.16034395]
 [  2.76691771   8.69698553]
 [  4.10669076   9.56604289]
 [  5.63997624  10.92410902]
 [  6.69124594  11.18521572]
 [  7.73722474  12.5186283 ]
 [  7.98813308  13.1652289 ]
 [ 12.28589376  12.28589376]
 [ 13.64771256  13.64771256]
 [ 12.15771275  16.34383593]
 [ 13.12680937  17.28061542]
 [ 14.47472953  17.649216  ]
 [ 16.60996408  20.47973035]
 [ 19.04441424  19.04441424]
 [ 16.76623655  21.7165282 ]
 [ 15.02476477  21.82468715]
 [ 19.20180101  24.70328373]
 [ 22.69502579  22.69502579]
 [ 21.76571324  26.87611803]
 [ 22.39200394  27.19098976]
 [ 23.36915823  27.31919674]
 [ 25.21579106  25.21579106]
 [ 25.          29.12707049]
 [ 25.97019113  29.99192169]
 [ 27.          31.09296771]
 [ 30.54882999  30.54882999]
 [ 29.23218572  33.36987373]
 [ 31.27440403  34.71557564]
 [ 32.17006362  35.18704729]
 [ 32.1990161   36.95284224]
 [ 35.29327451  35.29327451]
 [ 34.14694258  38.72800694]
 [ 34.60521016  39.08302186]
 [ 35.6964706   40.31564156]
 [ 37.06820749  41.41433662]
 [ 37.40083825  41.65781495]
 [ 40.89840471  40.89840471]
 [ 42.48261954  42.48261954]
 [ 39.55307932  45.62621215]
 [ 41.87826528  46.80941386]
 [ 45.01538644  45.01538644]
 [ 43.93249061  48.05863334]
 [ 45.18348089  49.36108848]
 [ 48.06458152  48.06458152]
 [ 47.3930598   51.75342423]
 [ 47.34368096  53.32032202]
 [ 50.88109501  50.88109501]
 [ 49.9311184   54.93551424]
 [ 52.97249908  52.97249908]
 [ 52.10812421  56.69009214]
 [ 54.84048566  54.84048566]
 [ 55.4550042   55.4550042 ]
 [ 56.70728165  56.70728165]
 [ 58.08825383  58.08825383]
 [ 59.61025997  59.61025997]
 [ 58.61102133  61.89351575]
 [ 61.83366234  61.83366234]
 [ 64.42291469  64.42291469]
 [ 64.88631311  64.88631311]
 [ 64.94770298  64.94770298]
 [ 64.96057404  64.96057404]
 [ 67.6019814   67.6019814 ]
 [ 67.93806597  67.93806597]
 [ 68.11465776  68.11465776]
 [ 67.96361145  67.96361145]
 [ 71.24769061  71.24769061]
 [ 71.46757732  71.46757732]
 [ 64.69161929  64.69161929]
 [ 64.74524721  64.74524721]
 [ 64.73871318  64.73871318]
 [ 72.57870905  76.34484898]
 [ 64.56005042  64.56005042]
 [ 73.8895437   73.8895437 ]
 [ 74.36059733  74.36059733]
 [ 76.6067987   76.6067987 ]
 [ 15.64823212  15.64823212]
 [ -0.44167596  -0.44167596]
 [ 38.12501941  38.12501941]
 [ -0.40119074  -0.40119074]
 [-66.03789588 -66.03789588]
 [ 38.13961855  38.13961855]
 [ -0.47826529  -0.47826529]
 [ 12.18958215  12.18958215]]';

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
% xlim([0 sam_ind(end)+1]); ylim([-30 30]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('MUSIC Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(ang_gt - ang_sam).^2;
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

sq_dif = abs(ang_gt - ang_sam_rm).^2;
RMSE_RMUSIC = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot the proposed CNN results
f_cnn(1) = figure(5);
plot(sam_ind, ang_gt(1,:),'Color','r');
hold on;
plot(sam_ind, ang_gt(2,:),'Color','b');
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
scatter(sam_ind, ang_gt(1,:)-CNN_pred(1,:),'Marker','d','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,ang_gt(2,:)-CNN_pred(2,:),'Marker','d','MarkerEdgeColor',[0 0 1]);
grid on;
% ylim([-150 150]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('CNN errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(ang_gt - CNN_pred).^2;
RMSE_CNN = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot l1-SVD results
f_gl1(1) = figure(7);
plot(sam_ind, ang_gt(1,:),'Color','r');
hold on;
plot(sam_ind, ang_gt(2,:),'Color','b');
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
scatter(sam_ind, ang_gt(1,:)-l1_SVD_ang_est(1,:),'Marker','*','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,ang_gt(2,:)-l1_SVD_ang_est(2,:),'Marker','*','MarkerEdgeColor',[0 0 1]);
grid on;
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('$\ell_{2,1}$-SVD errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(ang_gt - l1_SVD_ang_est).^2;
RMSE_l1_SVD = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot the MLP results
f_mlp(1) = figure(9);
plot(sam_ind, ang_gt(1,:),'Color','r');
hold on;
plot(sam_ind, ang_gt(2,:),'Color','b');
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
scatter(sam_ind, ang_gt(1,:)-ang_MLP(1,:),'Marker','x','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,ang_gt(2,:)-ang_MLP(2,:),'Marker','x','MarkerEdgeColor',[0 0 1]);
grid on;
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('MLP errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(ang_gt - ang_MLP).^2;
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
% ylim([-2.2 2.2]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('ESPRIT Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(ang_gt - ang_sam_esp).^2;
RMSE_ESPRIT = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

% Plot UnESPRIT results
f_unesp(1) = figure(13);
plot(sam_ind, ang_gt(1,:),'Color','r');
hold on;
plot(sam_ind, ang_gt(2,:),'Color','b');
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
scatter(sam_ind, ang_gt(1,:)-UnESPRIT_ang_est(1,:),'Marker','s','MarkerEdgeColor',[1 0 0]);
hold on;
scatter(sam_ind,ang_gt(2,:)-UnESPRIT_ang_est(2,:),'Marker','s','MarkerEdgeColor',[0 0 1]);
grid on;
% ylim([-2.2 2.2]);
legend('$\Delta \theta_1=\theta_1 - \hat{\theta}_1$','$\Delta \theta_2=\theta_2 - \hat{\theta}_2$','interpreter','latex');
title('UnESPRIT Errors', 'interpreter','latex');
ylabel('DoA [degrees]', 'interpreter','latex');
xlabel('Sample index', 'interpreter','latex');

sq_dif = abs(ang_gt - UnESPRIT_ang_est).^2;
RMSE_UnESPRIT = sqrt(sum(sq_dif(:))/length(sam_ind)/2);

VarNames = {'RMSE'};
Tab = table(VarNames, RMSE_MUSIC, RMSE_RMUSIC, RMSE_CNN, RMSE_l1_SVD,RMSE_MLP, RMSE_ESPRIT, RMSE_UnESPRIT)

