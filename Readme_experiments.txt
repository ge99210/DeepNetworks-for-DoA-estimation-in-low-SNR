Final Results

Author: Gergios K. Papageorgiou
Date: 14/06/2021

For use please cite the work:
...to do



MATLAB: (M)
PYTHON: (P)

Part I: training in the low-SNR regime from -20 to 0 db SNR for K=2 (25-55 min on TITAN RTX GPU) depending on the grid used (narrow or wider)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MATLAB (M) files stored in: 'D:MATLAB Drive\DoA_estimation_journal\Final'
PYTHON (P) CNN (proposed) files stored in (P1): 'D:...\TSP_JOURNAL_DoA_Estimation\DoA_Estimation_Journal'||| Trained weights in RQ: 'Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_Adam_dropRoP_0_7.h5' for fixed K=2 and narrow range; and 'Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_RoP_0_7_90deg_v6c.h5' for fixed K=2 and wide range; and 'Model_CNN_DoA_N16_K1to3_res1_min15to0dBSNR_Kunknown_adam_bs32_lr1emin3.h5' for mixed K=1-3 and narrow range.
PYTHON (P) MLP files stored in (P2): 'D:OneDrive - Heriot-Watt University\DoA DATA\Array Imperfections Model\DNN-DOA-master_original'

Location where the training/tetsing data and l_21-SVD, UnESPRIT results are stored (LR1): 'D:.../DoA_DATA_JOURNALS'
Location where the MUSIC, R-MUSIC and ESPRIT results are stored (LR2): 'D:...\DoA_Estimation_underdetermined\ComparisonRESULTS'
Location where CNN parameters are stored (LR3): 'C:...\DoA_Estimation_Journal' - ZWS Edinburgh

GENERATE Training Data (M): Run 'GENER_Train_Data_DoA_JOURNAL_3D.m' and save as 'TRAIN_DATA_16ULA_K2_low_SNR_res1_3D.h5' or 'TRAIN_DATA_16ULA_K2_low_SNR_res1_3D_90deg.h5' (depending on the grid narrow or wide)

TRAIN the CNN (P): Run
1) fixed K, narrow grid: 'CNN_training_lowSNR_new_training_RQ_test.py' and save weights/architecture in LR3
2) fixed K, wide grid: 'CNN_training_lowSNR_new_training_RQ_test_90deg.py' and save weights/architecture in LR3
3) mixed K, narrow grid:  'CNN_training_allSNR_multipleK_unknownK.py' for mixed K and SNR -15 dB to 0 dB and save the weights as 'Model_CNN_DoA_N16_K1to3_res1_min15to0dBSNR_Kunknown_adam_bs32_lr1emin3.h5' in LR3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Experiment set 1: Fixed SNR - robustness across offgrid angles

A) Setup/parameters: 
SNR: -10dB 
Snapshots: T=2000
Δθ=4.7 degrees
Angles: first signal slides from -60:1:55 and the second is these +Δθ
Test set size: 116 examples
RMSE: MUSIC 5.01, R-MUSIC 0.35, CNN 0.30, l1_SVD 1, ESPRIT 0.35, UnESPRIT 0.27

RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp1A.m' and save as 'TEST_DATA_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5' in LR1

ii) l1-SVD results (M): saved by running i) as 'RMSE_l1SVD_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5' (threshold value eta=550)

iii) UnESPRIT (M): saved by running i) as 'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5' (ds=1, ms=8)

iv) Test MUSIC, R-MUSIC, ESPRIT (M): run 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp1A.m' (loads the testing data from i) and saves as 'Slide_angsep4coma7_K2_min10dB_T2000.mat' in LR2

v) Test the CNN (P): 'CNN_testing_Exp1A_new_training.ipynb' and saves the results as 'Slide_angsep4coma7_K2_min10dB_T2000_CNN_new_RQ.h5' in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.) and 'Exp1A.py' to test it. Results saved in P2 as DoA_est_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results.  

vii) Plot the results (M): Run 'plot_experiment1A.m' (after loading the corresponding folders)-figures saved in \RESULTS


B) Setup/parameters: 
SNR: 0dB 
Snapshots: T=200
Δθ=2.11 degrees
Angles: first signal slides from -59.5:1:57.5 and the second is these +Δθ
Test set size: 118 examples
RMSE: MUSIC 20.31, R-MUSIC 11.17, CNN 0.50, l1_SVD 0.54, ESPRIT 1.23, UnESPRIT 0.83

RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp1B.m' and save as 'TEST_DATA_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11.h5' in LR1

ii) l1-SVD results (M): saved by running i) as 'RMSE_l1SVD_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11.h5' (threshold value eta=60)

iii) UnESPRIT (M): saved by running i) as 'RMSE_UnESPRIT_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11.h5' (ds=1, w=8)

iv) Test MUSIC, R-MUSIC, ESPRIT (M): Run 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp1B.m' (loads the testing data in i) and saves as 'Slide_angsep2coma11_K2_0dB_T200.mat' in LR2

v) Test the CNN (P): 'CNN_testing_Exp1B_new_training.ipynb' and saves the data as 'Slide_ang_2coma11sep_K2_0dB_T200_CNN_RQ.h5' in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.) and 'Exp1B.py' to test it. Results saved in P2 as DoA_est_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results.  

vii) Plot the results (M): Run 'plot_experiment1B.m' (after loading the corresponding folders)-figures saved in \RESULTS


C) Setup/parameters for wide range [-90,90]:
SNR: -10dB 
Snapshots: T=2000
Δθ=4.7 degrees
Angles: first signal slides from -90:1:85 and the second is these +Δθ
Test set size: 176 examples
RMSE: MUSIC 50.43, R-MUSIC 32.22, CNN 0.96, l1_SVD 11.73, ESPRIT 30.12, UnESPRIT 26.49

RUN:
i) Generate Testing Data (M): change SOURCE.interval = 90; and run 'GENER_Test_Data_DoA_JOURNAL_3D_Exp1A.m'. Save as 'TEST_DATA_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7_90deg.h5' in LR1

ii) l1-SVD results (M): saved by running i) as 'RMSE_l1SVD_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7_90deg.h5' (threshold value eta=550)

iii) UnESPRIT (M): saved by running i) as 'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7_90deg.h5' (ds=1, w=8)

iv) Test MUSIC, R-MUSIC, ESPRIT (M): run 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp1C.m' (loads the testing data from i) and saves as 'Slide_angsep4coma7_K2_min10dB_T2000_90deg.mat' in LR2

v) Test the CNN (P): 'CNN_testing_Exp1C_new_training_90deg.ipynb' and saves the results as 'Slide_angsep4coma7_K2_min10dB_T2000_CNN_new_RQ_10drop.h5' (or 'Slide_angsep4coma7_K2_min10dB_T2000_CNN_new_RQ_90deg_vf6c.h5'-newest) in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.) and 'Exp1A.py' to test it (load the test data in i). Results saved in P2 as DoA_est_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results.  

vii) Plot the results (M): Run 'plot_experiment1C_90deg.m' (after loading the corresponding folders)-figures saved in \RESULTS

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Experiment set 2: RMSE vs SNR on fixed off-grid angles

Setup/parameters: 
Snapshots: T=1000
Angles: 10.11, 13.3 degrees (Δθ=3.19)

RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp2.m' and save as 'TEST_DATA1K_16ULA_K2_fixed_offgrid_ang_allSNR_T1000_3D.h5' in LR1

ii) l1-SVD RMSE results (without saving the y data) (M): saved with i) in the same file with threshold value eta_vec = [1260 700 400 230 140 100 70 70 60 60 60]

iii) UnESPRIT (M): saved by running i) in the same file (ds=1, w=8)

iv) Test MUSIC, R-MUSIC, ESPRIT + CRLB, CRLB_uncr (M): 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp2.m' and save the results as'RMSE_K2_offgrid_ang_fixed_allSNR_T1000.mat' in LR2

v) Test the CNN (P): 'Test_CNN_RMSE_results_Exp2_low_training.ipynb' and save the results as 'RMSE_CNN_K2_fixed_offgrid_ang_all_SNR_T1000_RQ.h5' in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.) and 'Exp2_fixed_set.py' to test it. Results saved in P2 as rmse_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results.  

vii) Plot the results (M): Run 'plot_experiment2.m' (after loading the corresponding folders)-figures saved in \RESULTS

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Experiment 3: RMSE vs T

Setup/parameters: 
Angles: -13.18 and -9.58 (Δθ=3.6) [degrees]
SNR = -10 dB 
T_vec = 1000*[0.1 0.2 0.5 1 2 5 10]

RUN: 
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp3.m' and save the data as 'TEST_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5' in LR1

ii) Save l1-SVD RMSE data (no y data saved) (M): runs with i) and saved (same folder) as 'RMSE_l1_SVD_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5' with threshold value eta_vec = [130 180 270 410 570 910 1280]

iii) UnESPRIT (M): saved by running i) in the same file (ds=1, w=8)

iv) Test MUSIC, R-MUSIC, ESPRIT + CRLB, CRLB_uncr (M): 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp3.m' and store data as 'RMSE_K2_min10dB_ang_sep3coma6_vsT.mat' in LR2

v) Test the CNN (P): 'Test_CNN_RMSE_results_Exp3_low_training.ipynb' and save the results as 'RMSE_CNN_K2_min10dBSNR_vsT_ang_sep3coma6_new_train_low_vf.h5' in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.) and 'Exp3_fixed_set.py' to test it. Results saved in P2 as rmse_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results.*** if the angles are not resolved the results are not included  

vii) Plot the results (M): Run 'plot_experiment3.m' (after loading the corresponding folders)-figures saved in \RESULTS

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Experiment 4: RMSE vs Δθ

Setup/parameters: 
T = 500
SNR=-10 dB
θ_1 = -13.8 [degrees] and θ_2 = θ_1 + Δθ

RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp4.m' and save the data as 'TEST_DATA1K_16ULA_K2_min10dBSNR_T500_3D_vs_ang_sep.h5' in LR1

ii) Save l1-SVD RMSE data (no y data saved) (M): runs with i) and save (same folder) as 'RMSE_l1_SVD_16ULA_K2_min10dBSNR_T500_3D_vs_ang_sep.h5' with threshold value eta = 290

iii) UnESPRIT (M): saved by running i) in the same file (ds=1, w=8)

iv) Test MUSIC, R-MUSIC, ESPRIT (M): 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp4.m' and save the results as 'RMSE_K2_vs_ang_sep_T500_min10dBSNR.mat' in LR2

v) Test the CNN (P): 'Test_CNN_RMSE_vs_ang_sep_Exp4_low_training.ipynb' and save the results as 'RMSE_CNN_K2_vs_ang_sep_T500_min10dBSNR_new.h5' in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.)[trained model with weights extension ..._Exp4] and 'Test_Exp4.py' to test it. Results saved in P2 as rmse_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results.*** if the angles are not resolved the results are not included 

vii) Plot the results (M): Run 'plot_experiment4.m' (after loading the corresponding folders)-figures saved in \RESULTS

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Experiment set 5: SNR mismatch - robustness 

A) Setup/parameters: 
SNR: 0dB (theoretical value assuming sigma_1^2 = sigma_2^2=1)
TX power: sigma_1^2 = 0.7, sigma_2^2 = 1.25 (actual SNR=-1.549)
Snapshots: T=200
Δθ=2.11 degrees
Angles: first signal slides from -59.5:1:57.5 and the second is these +Δθ
Test set size: 118 examples
RMSE: MUSIC 21.46, R-MUSIC 15.79, CNN 0.46, l1_SVD 0.70, ESPRIT 1.49, UnESPRIT 0.91

RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp5A.m' and save data as 'TEST_DATA_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11_power_mismatch.h5' in LR1 

ii) l1-SVD results (M): also saved by running i) as 'RMSE_l1SVD_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11_power_mismatch.h5' in LR1 (threshold value eta=60 - as in Exp.1B)

iii) UnESPRIT (M): saved by running i) in the same location as 'RMSE_UnESPRIT_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11_power_mismatch.h5' (ds=1, w=8)

iv) Test MUSIC, R-MUSIC, ESPRIT (M): 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp5A.m' and save the results as 'Slide_angsep2coma11_K2_0dB_T200_power_mismatch.mat' in LR2

v) Test the CNN (P): 'CNN_testing_Exp5A_new_training.ipynb' and save the data as 'Slide_ang_2coma11sep_K2_0dB_T200_CNN_new_power_mismatch_new.h5' in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.) or use the trained weights from Exp1B. Run 'Exp5A.py' to test it. Results saved in P2 as DoA_est_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results. 

vii) Plot the results (M): Run 'plot_experiment5A.m' (after loading the corresponding folders)-figures saved in \RESULTS

B) Setup/parameters: 
SNR: -10dB (theoretical value assuming sigma_1^2 = sigma_2^2=1)
TX power: sigma_1^2 = 0.7, sigma_2^2 = 1.25 (actual SNR=-11.549)
Snapshots: T=1000
Δθ=4 degrees
Angles: first signal slides from -59.43:1:55.57 and the second is these +Δθ
Test set size: 116 examples
RMSE: MUSIC 20.26, R-MUSIC 15.85, CNN 0.76, l1_SVD 1.42, ESPRIT 1.49, UnESPRIT 1.08

RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp5B.m' and save data as 'TEST_DATA_16ULA_K2_min10dBSNR_T1000_3D_slideang_offgrid_sep4_power_mismatch.h5' in LR1 

ii) l1-SVD results (M): also saved by running i) as 'RMSE_l1SVD_16ULA_K2_min10dBSNR_T1000_3D_slideang_offgrid_sep4_power_mismatch.h5' in LR1 (threshold value eta=400 - optimized)

iii) UnESPRIT (M): saved by running i) in the same location as 'RMSE_UnESPRIT_16ULA_K2_min10dBSNR_T1000_3D_slideang_offgrid_sep4_power_mismatch.h5' (ds=1, w=8)

iv) Test MUSIC, R-MUSIC, ESPRIT (M): 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp5B.m' and save the results as 'Slide_angsep4_K2_min10dB_T1000_power_mismatch.mat' in LR2

v) Test the CNN (P): 'CNN_testing_Exp5B_new_training.ipynb' and save the data as 'Slide_angsep4_K2_min10dB_T1000_CNN_new_power_mismatch_new_lr_RoP_0_7.h5' in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.) or use the trained weights from Exp1A. Run 'Exp5B.py' to test it. Results saved in P2 as DoA_est_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results. 

vii) Plot the results (M): Run 'plot_experiment5B.m' (after loading the corresponding folders)-figures saved in \RESULTS

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Experiment set 6: Correlated signals (RQ) 

Setup/parameters: 
SNR: -10dB 
TX power: sigma_1^2 = 1, sigma_2^2 = 1
Snapshots: T=1000
Δθ = 3.8 degrees
Angles: theta1 = 10, theta2 = 13.8 
rho_vec = 0:0.1:1 
P = [1 rho; rho 1]; Pc = [1 0; rho sqrt(1-rho^2)]; Sc = Pc*S;

i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp6.m' and save data as 'TEST_DATA1K_3D_16ULA_K2_fixed_ang_sep3coma8_min10dBSNR_T1000_vsrho.h5' in LR1 

ii) l1-SVD results (M): also saved by running i) in LR1 (threshold value eta=400)

iii) UnESPRIT (M): saved by running i) in the same location and file name (ds=1, w=8)

iv) Test MUSIC, R-MUSIC, ESPRIT (M): 'MUSIC_RMUSIC_ESPRIT_TESTING_3D_Exp6.m' and save the results as 'RMSE_K2_offgrid_ang_fixed_min10dBSNR_T1000_vsrho.mat' in LR2

v) Test the CNN (P): 'Test_CNN_RMSE_results_Exp6_vs_correl_coeff.ipynb' and save the data as 'RMSE_CNN_K2_fixed_offgrid_ang3coma8_min10dBSNR_T1000_vsrho_lrRoP_0_7.h5' in LR2

vi) Train and test the MLP (P): 'main_ArrayImperfections_GP.py' to train the network (in the selected grid, SNR, etc.) or use the trained weights from Exp1A. Run 'Exp6_rho.py' to test it. Results saved in P2 as DoA_est_path = '... .npy', but need to run 'Read_DoA_Estimates.ipynb' in jupyter notebook to get the results.*** if angles not resolved RMSE not included.

vii) Plot the results (M): Run 'plot_experiment6.m' (after loading the corresponding folders)-figures saved in \RESULTS

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Part II: training with more sources (K_max=3) at random SNRs with K prediction 

GENERATE Training Data (M): Run 'GENER_Train_Data_DoA_JOURNAL_3D_multipleK.m' and save the train data as 'TRAIN_DATA_16ULA_K1to3_min20to0dBSNR_res1_3D.h5' in LR1

TRAIN the CNN (P): run 'CNN_training_allSNR_multipleK_unknownK.ipynb' for mixed K and SNR -15 dB to 0 dB and save the weights as 'Model_CNN_DoA_N16_K1to3_res1_min15to0dBSNR_Kunknown_adam_bs32_lr1emin3.h5'

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Experiment set 7: Unknown K - mixed number of sources

A) Setup/parameters: 
SNR: -10dB
Snapshots: T=3000
ΔΘ = 5.2[degrees]
Sources 1 to K_max=3
1st signal angle: -7.8 
2nd signal angle: -7.8+Δθ = -2.6
3rd signal angle: -7.8+2*Δθ = 2.6
Nsim = 10,000

CNN results [degrees]:
K=1 \bar{p}=0.9
μ(d): \infty  
max(d): \infty
(K known) RMSE=0.2

K=2 \bar{p}=0.8
μ(d): \infty 
max(d): \infty
(K known) RMSE=0.44

K=3 \bar{p}=0.6
μ(d): 1.42  
max(d): 10.8
(K known) RMSE=0.61


RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp7A.m' stored in LR1

'TEST_DATA_16ULA_K1_min10dBSNR_T3000_3D_fixedang_offgrid.h5' 
'TEST_DATA_16ULA_K2_min10dBSNR_T3000_3D_fixedang_offgrid.h5' 
'TEST_DATA_16ULA_K3_min10dBSNR_T3000_3D_fixedang_offgrid.h5' 

ii) Test the CNN (P): 'CNN_testing_Exp7A_allSNR_test_min10dB.py' and store the data as 'Exper6_conf_mat_min10dB_RQ.eps' in LR3 


SNR: 0dB
Snapshots: T=1000
ΔΘ = 5.2[degrees]
Sources 1 to K_max=3
1st signal angle: -7.8 
2nd signal angle: -7.8+Δθ = -2.6
3rd signal angle: -7.8+2*Δθ = 2.6
Nsim = 10,000

CNN results [degrees]:
K=1 \bar{p}=0.9
μ(d): 0.2  
max(d): 0.2
(K known) RMSE=0.2

K=2 \bar{p}=0.8
μ(d): 0.61
max(d): 5.4
(K known) RMSE=0.44

K=3 \bar{p}=0.6
μ(d): 1.38 
max(d): 10.8
(K known) RMSE=0.66

RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp7A.m' stored in LR1

'TEST_DATA_16ULA_K1_0dBSNR_T1000_3D_fixedang_offgrid.h5' 
'TEST_DATA_16ULA_K2_0dBSNR_T1000_3D_fixedang_offgrid.h5' 
'TEST_DATA_16ULA_K3_0dBSNR_T1000_3D_fixedang_offgrid.h5' 

ii) Test the CNN (P): 'CNN_testing_Exp7A_allSNR_test_0dB.py' and store the data as 'Exper6_conf_mat_0dB_RQ.eps' in LR3 


B) Setup/parameters: 

SNR: -10dB
Snapshots: T=3,000
ΔΘ = 10 [degrees]
Sources 1 to K_max=3
K=1 1st signal angle: -59.8:+1:59.2 (120 examples)
K=2 1st signal angle: -59.8:+1:49.2, 2nd signal angle: -49.8:+1:59.2 (110 examples) [+ΔΘ]
K=3 1st signal angle: -59.8:+1:39.2, 2nd signal angle: -49.8:+1:49.2, 3rd signal angle: -39.8:+1:59.2 (100 examples) [+2*ΔΘ]

CNN results:
K=1 \bar{p}=0.8
μ(d):  0.21
max(d): 0.8

K=2 \bar{p}=0.8
μ(d): 0.30
max(d): 9.8

K=3 \bar{p}=0.8
μ(d): 0.54 
max(d): 10.8


RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp7B.m' stored in LR1

'TEST_DATA_16ULA_K1_min10dBSNR_T3000_3D_slideang_offgrid_ang_sep10.h5' 
'TEST_DATA_16ULA_K2_min10dBSNR_T3000_3D_slideang_offgrid_ang_sep10.h5' 
'TEST_DATA_16ULA_K3_min10dBSNR_T3000_3D_slideang_offgrid_ang_sep10.h5' 

ii) Test the CNN (P): 'CNN_testing_Exp7B_min10dB_unknownK.py' and store the data as 'Slide_ang_sep10_K1to3_min10dB_T3000_CNN_RQ.h5' in LR2
iii) Plot the results in Matlab using 'plot_experiment7B_SNRmin10dB.m'

SNR: 0dB
Snapshots: T=1,000
ΔΘ = 10 [degrees]
Sources 1 to K_max=3
K=1 1st signal angle: -59.8:+1:59.2 (120 examples)
K=2 1st signal angle: -59.8:+1:49.2, 2nd signal angle: -49.8:+1:59.2 (110 examples) [+ΔΘ]
K=3 1st signal angle: -59.8:+1:39.2, 2nd signal angle: -49.8:+1:49.2, 3rd signal angle: -39.8:+1:59.2 (100 examples) [+2*ΔΘ]

CNN results:
K=1 \bar{p}=0.8
μ(d):  \infty
max(d): \infty

K=2 \bar{p}=0.8
μ(d): 0.4
max(d): 10.2

K=3 \bar{p}=0.8
μ(d): 0.33  
max(d): 9.8


RUN:
i) Generate Testing Data (M): 'GENER_Test_Data_DoA_JOURNAL_3D_Exp7B.m' stored in LR1

'TEST_DATA_16ULA_K1_0dBSNR_T1000_3D_slideang_offgrid_ang_sep10.h5' 
'TEST_DATA_16ULA_K2_0dBSNR_T1000_3D_slideang_offgrid_ang_sep10.h5' 
'TEST_DATA_16ULA_K3_0dBSNR_T1000_3D_slideang_offgrid_ang_sep10.h5' 

ii) Test the CNN (P): 'CNN_testing_Exp7B_0dB_unknownK.py' and store the data as 'Slide_ang_sep10_K1to3_0dB_T1000_CNN_RQ.h5' in LR2
iii) Plot the results in Matlab using 'plot_experiment7B_SNR0dB.m'

