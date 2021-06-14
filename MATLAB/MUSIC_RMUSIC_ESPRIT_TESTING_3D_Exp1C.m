% DAE Testing 
% Georgios K. Papageorgiou 03/02/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
% clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
    'TEST_DATA_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7_90deg.h5');
h5disp(filename);
r_sam = h5read(filename, '/sam');
R_sam = squeeze(r_sam(:,:,1,:)+1j*r_sam(:,:,2,:));
r_the = h5read(filename, '/theor');
R_the = squeeze(r_the(:,:,1,:)+1j*r_the(:,:,2,:));
True_angles = h5read(filename, '/angles');
SOURCE_K = size(True_angles,1);
[ULA_N,~, N_test] = size(R_sam);
SOURCE.interval = 90;
res = 1;
T = 2000;
% UnESPRIT pars 
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
ms = 8; % if 1 the weights are equal if ms>1 there are higher weights at the center elements of each subarray
w = min(ms,ULA_N-ds-ms+1);  % Eq 9.133 in [1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialization
rmse_the = 0;
rmse_sam = 0;
rmse_the_rm = 0;
rmse_sam_rm = 0;
rmse_the_esp = 0;
rmse_sam_esp = 0;
rmse_the_unesp = 0;
rmse_sam_unesp = 0;


ang_the = zeros(SOURCE_K,N_test);
ang_sam = zeros(SOURCE_K,N_test);
ang_the_rm = zeros(SOURCE_K,N_test);
ang_sam_rm = zeros(SOURCE_K,N_test);
ang_gt = zeros(SOURCE_K,N_test);
ang_the_esp = zeros(SOURCE_K,N_test);
ang_sam_esp = zeros(SOURCE_K,N_test);
ang_the_unesp = zeros(SOURCE_K,N_test);
ang_sam_unesp = zeros(SOURCE_K,N_test);


for nit=1:N_test
   
   % The true covariance matrix
   Rx = R_the(:,:,nit);
    
   % The smoothed sample covariance matrix
   Rx_sam = R_sam(:,:,nit);
    
   %% MUSIC estimator 
   [doas_the, spec_the, specang_the] = musicdoa(Rx,SOURCE_K,'ScanAngles', -SOURCE.interval:res:SOURCE.interval);
   [doas_sam, spec_sam, specang_sam] = musicdoa(Rx_sam,SOURCE_K, 'ScanAngles', -SOURCE.interval:res:SOURCE.interval);
  
   ang_the(:,nit) = sort(doas_the)';
   ang_sam(:,nit)  = sort(doas_sam)';
   ang_gt(:,nit) = sort(True_angles(:,nit));
   
   % RMSE calculation
   rmse_the = rmse_the + norm(ang_the(:,nit) - ang_gt(:,nit))^2;
   rmse_sam = rmse_sam + norm(ang_sam(:,nit) - ang_gt(:,nit))^2;
      
   %% Root-MUSIC estimator 
   doas_the_rm = sort(rootmusicdoa(Rx, SOURCE_K))';
   doas_sam_rm = sort(rootmusicdoa(Rx_sam, SOURCE_K))';

   ang_the_rm(:,nit) = sort(doas_the_rm);
   ang_sam_rm(:,nit) = sort(doas_sam_rm);
   
   % RMSE calculation - degrees
   rmse_the_rm = rmse_the_rm + norm(ang_the_rm(:,nit) - ang_gt(:,nit))^2;
   rmse_sam_rm = rmse_sam_rm + norm(ang_sam_rm(:,nit) - ang_gt(:,nit))^2;
   
    %% ESPRIT (with variable ds and reweighting technique)   
   % EPSRIT
   
   doas_the_esp = ESPRIT_doa(Rx, ds, SOURCE_K, w);
   doas_sam_esp = ESPRIT_doa(Rx_sam, ds, SOURCE_K, w);
     
   ang_the_esp(:,nit) = sort(doas_the_esp);
   ang_sam_esp(:,nit) = sort(doas_sam_esp);
   
   % ang = espritdoa(Rx_sam,SOURCE_K);
   rmse_the_esp = rmse_the_esp + norm(ang_the_esp(:,nit) - ang_gt(:,nit))^2;
   rmse_sam_esp = rmse_sam_esp + norm( ang_sam_esp(:,nit)- ang_gt(:,nit))^2;
%    %% Unitary ESPRIT (with variable ds and reweighting technique)   
%    % Unitary ESPRIT
%    doas_unit_ESPRIT_the = unit_ESPRIT(X, T, ds, SOURCE_K, w);
%    doas_unit_ESPRIT_sam = unit_ESPRIT(Y, T, ds, SOURCE_K, w);
%    
%    ang_the_unesp(:,nit) = sort(doas_unit_ESPRIT_the);
%    ang_sam_unesp(:,nit) = sort(doas_unit_ESPRIT_sam);
%    
%    rmse_the_unit_esp = rmse_the_unit_esp + norm(ang_the_unesp(:,nit) - ang_gt(:,nit))^2;
%    rmse_sam_unit_esp = rmse_sam_unit_esp + norm(ang_sam_unesp(:,nit) - ang_gt(:,nit))^2;
        
   nit
end

% MUSIC RMSE_deg
RMSE_the = sqrt(rmse_the/SOURCE_K/N_test);
RMSE_sam = sqrt(rmse_sam/SOURCE_K/N_test);
MUSIC_deg = [ RMSE_the; RMSE_sam];

% R-MUSIC RMSE_deg
RMSE_the_rm = sqrt(rmse_the_rm/SOURCE_K/N_test);
RMSE_sam_rm = sqrt(rmse_sam_rm/SOURCE_K/N_test);
R_MUSIC_deg = [ RMSE_the_rm; RMSE_sam_rm];

% ESPRIT_deg
RMSE_the_esp = sqrt(rmse_the_esp/SOURCE_K/N_test);
RMSE_sam_esp = sqrt(rmse_sam_esp/SOURCE_K/N_test);
ESPRIT_deg = [ RMSE_the_esp; RMSE_sam_esp];

% % UnESPRIT_deg
% RMSE_the_unesp = sqrt(rmse_the_unit_esp/SOURCE_K/N_test);
% RMSE_sam_unesp = sqrt(rmse_sam_unit_esp/SOURCE_K/N_test);
% R_UnESPRIT_deg = [ RMSE_the_unesp; RMSE_sam_unesp];

VarNames = {'Theoretical'; 'Sampled'};
Tab = table(VarNames, MUSIC_deg, R_MUSIC_deg,ESPRIT_deg)

% % Save the results 
save_path = 'C:\Users\geo_p\OneDrive - Heriot-Watt University\HWU2\Code\Python\DoA Estimation\DoA_Estimation_underdetermined\ComparisonRESULTS';
save(fullfile(save_path,'Slide_angsep4coma7_K2_min10dB_T2000_90deg.mat'),'MUSIC_deg','R_MUSIC_deg','ESPRIT_deg','ang_the','ang_sam','ang_the_rm','ang_sam_rm','ang_the_esp','ang_sam_esp','ang_gt');
