#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, Softmax
from keras.regularizers import l2, l1
from keras.initializers import glorot_normal
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras


# In[2]:


# New model with low-SNR training - 25/09/2020
model_CNN = load_model('Model_CNN_DoA_class_Data_N16_K1to3_res1_0dBSNR_v2.h5') 
# model_CNN = load_model('Model_CNN_DoA_N16_K1to3_res1_min15to0dBSNR_Kunknown_adam_bs64_lr1emin3.h5')


# In[3]:


res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)


# In[4]:


# Load the Test Data for the Experiment
# K=1 - 120 examples
filename1 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K1_0dBSNR_T1000_3D_slideang_offgrid_ang_sep10.h5'
f1 = h5py.File(filename1, 'r')
GT_angles1 = np.array(f1['angles'])
Ry_the_test1 = np.array(f1['the'])
Ry_sam_test1 = np.array(f1['sam']) 
Nsim1 = GT_angles1.shape[0]
# K=2 - 110 examples
filename2 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K2_0dBSNR_T1000_3D_slideang_offgrid_ang_sep10.h5'
f2 = h5py.File(filename2, 'r')
GT_angles2 = np.array(f2['angles'])
Ry_the_test2 = np.array(f2['the'])
Ry_sam_test2 = np.array(f2['sam']) 
Nsim2 = GT_angles2.shape[0]
# K=3 - 110 examples
filename3 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K3_0dBSNR_T1000_3D_slideang_offgrid_ang_sep10.h5'
f3 = h5py.File(filename3, 'r')
GT_angles3 = np.array(f3['angles'])
Ry_the_test3 = np.array(f3['the'])
Ry_sam_test3 = np.array(f3['sam']) 
Nsim3 = GT_angles3.shape[0]


# In[5]:


Ry_sam_test1.shape


# In[6]:


# First permute the tensor and then predict
[n_test,dim,N,M]=Ry_sam_test1.shape
X_test_data_sam1 = Ry_sam_test1.swapaxes(1,3)
X_test_data_sam2 = Ry_sam_test2.swapaxes(1,3)
X_test_data_sam3 = Ry_sam_test3.swapaxes(1,3)

X_test_data_the1 = Ry_the_test1.swapaxes(1,3)
X_test_data_the2 = Ry_the_test2.swapaxes(1,3)
X_test_data_the3 = Ry_the_test3.swapaxes(1,3)

X_test_data_sam1.shape


# In[7]:


# K =1 with Sample Covariance Estimate
B1 = GT_angles1
K = 1
# Run inference on CPU
with tf.device('/cpu:0'):
    # Estimation with the covariance matrix
    x_pred_sam1 = model_CNN.predict(X_test_data_sam1)
# Classify K sources with maximum probability - Scenario 1
x_ind_K_sam1 = np.argpartition(x_pred_sam1, -K, axis=1)[:, -K:]
A_sam1 = np.sort(v[x_ind_K_sam1])
# Calculate the RMSE [in degrees]
RMSE_sam1 = round(np.sqrt(mean_squared_error(np.sort(A_sam1), np.sort(B1))),4)
print(RMSE_sam1)


# In[8]:


# K =1 with True Covariance Matrix
K = 1
# Run inference on CPU
with tf.device('/cpu:0'):
    # Estimation with the covariance matrix
    x_pred_the1 = model_CNN.predict(X_test_data_the1)
# Classify K sources with maximum probability - Scenario 1
x_ind_K_the1 = np.argpartition(x_pred_the1, -K, axis=1)[:, -K:]
A_the1 = np.sort(v[x_ind_K_the1])
# Calculate the RMSE [in degrees]
RMSE_the1 = round(np.sqrt(mean_squared_error(np.sort(A_the1), np.sort(B1))),4)
print(RMSE_the1)


# In[9]:


# K =2
B2 = GT_angles2
K = 2
# Run inference on CPU
with tf.device('/cpu:0'):
    # Estimation with the covariance matrix
    x_pred_sam2 = model_CNN.predict(X_test_data_sam2)
# Classify K sources with maximum probability - Scenario 1
x_ind_K_sam2 = np.argpartition(x_pred_sam2, -K, axis=1)[:, -K:]
A_sam2 = np.sort(v[x_ind_K_sam2])
# Calculate the RMSE [in degrees]
RMSE_sam2 = round(np.sqrt(mean_squared_error(np.sort(A_sam2), np.sort(B2))),4)
print(RMSE_sam2)


# In[10]:


# K =2
K = 2
# Run inference on CPU
with tf.device('/cpu:0'):
    # Estimation with the covariance matrix
    x_pred_the2 = model_CNN.predict(X_test_data_the2)
# Classify K sources with maximum probability - Scenario 1
x_ind_K_the2 = np.argpartition(x_pred_the2, -K, axis=1)[:, -K:]
A_the2 = np.sort(v[x_ind_K_the2])
# Calculate the RMSE [in degrees]
RMSE_the2 = round(np.sqrt(mean_squared_error(np.sort(A_the2), np.sort(B2))),4)
print(RMSE_the2)


# In[11]:


# K =1
B3 = GT_angles3
K = 3
# Run inference on CPU
with tf.device('/cpu:0'):
    # Estimation with the covariance matrix
    x_pred_sam3 = model_CNN.predict(X_test_data_sam3)
# Classify K sources with maximum probability - Scenario 1
x_ind_K_sam3 = np.argpartition(x_pred_sam3, -K, axis=1)[:, -K:]
A_sam3 = np.sort(v[x_ind_K_sam3])
# Calculate the RMSE [in degrees]
RMSE_sam3 = round(np.sqrt(mean_squared_error(np.sort(A_sam3), np.sort(B3))),4)
print(RMSE_sam3)


# In[12]:


# K =3
K = 3
# Run inference on CPU
with tf.device('/cpu:0'):
    # Estimation with the covariance matrix
    x_pred_the3 = model_CNN.predict(X_test_data_the3)
# Classify K sources with maximum probability - Scenario 1
x_ind_K_the3 = np.argpartition(x_pred_the3, -K, axis=1)[:, -K:]
A_the3 = np.sort(v[x_ind_K_the3])
# Calculate the RMSE [in degrees]
RMSE_the3 = round(np.sqrt(mean_squared_error(np.sort(A_the3), np.sort(B3))),4)
print(RMSE_the3)


# In[13]:


# Check here the threshold prediction without assuming knowledge of K, which is here used only for the rate calculation
def noK_prediction_slide(x_pred_sam, prob, K, GT_angles): 
    log = (x_pred_sam >= prob).astype(int)
    Dsz = log.shape[0]
    KMat = log.sum(axis=1)
    Kmax = max(KMat)
    # probability *%
    Corp = sum(KMat==K)*100/Dsz
    Fap = sum(KMat>K)*100/Dsz
    Mdp = sum(KMat<K)*100/Dsz
    A_sam = np.zeros((Dsz,Kmax))
    x = float("nan")
    count = 0
    sum_rmse = 0
    for n in range(Dsz):
        val = v[log[n]==1] 
        if val.size==K:
            count += 1
            sum_rmse += np.linalg.norm(np.sort(GT_angles[n]) - np.sort(val))**2  
        if val.size<Kmax:
            val = np.append(val,np.tile(x,(Kmax-val.size)))          
        A_sam[n] = val
    RMSE = np.sqrt(sum_rmse/K/count)    
    return A_sam, Corp, Fap, Mdp, KMat, RMSE  


# In[14]:


# Optimize the probabilities by chosing the best prob. threshold value 
prob_vec = np.linspace(0.4,1,num=61)
Corp_vec = np.zeros((prob_vec.size,1))
Fap_vec = np.zeros((prob_vec.size,1))
Mdp_vec = np.zeros((prob_vec.size,1))
for n in range(prob_vec.size):
    A_sam_thresh_vec, Corp_vec[n], Fap_vec[n], Mdp_vec[n],KMat, RMSE = noK_prediction_slide(x_pred_sam1, prob_vec[n], 1, GT_angles1)
ind1a = np.argmax(Corp_vec)
# Plot the results of the detection
plt.plot(prob_vec,Corp_vec,label='Correct Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec[ind1a])


# In[15]:


ind1b = np.argmin(Mdp_vec)
plt.plot(prob_vec,Fap_vec,label='FA Prob.')
plt.plot(prob_vec,Mdp_vec,label='MD Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec[ind1b])


# In[18]:


prob1 = 0.9
A_sam_thresh1, Corp1, Fap1, Mdp1,KMat1, RMSE1 = noK_prediction_slide(x_pred_sam1, prob1, 1, B1)
print('The probability of detecting K=1 source is:', Corp1,'% and RMSE =',RMSE1,'for K=1') 
print('The false alarm probability is:',Fap1,'%') 
print('The misdetection probability is:',Mdp1,'%')


# In[19]:


# Optimize the probabilities by chosing the best prob. threshold value 
prob_vec = np.linspace(0.4,1,num=61)
Corp_vec = np.zeros((prob_vec.size,1))
Fap_vec = np.zeros((prob_vec.size,1))
Mdp_vec = np.zeros((prob_vec.size,1))
for n in range(prob_vec.size):
    A_sam_thresh_vec, Corp_vec[n], Fap_vec[n], Mdp_vec[n],KMat, RMSE = noK_prediction_slide(x_pred_sam2, prob_vec[n], 2, GT_angles2)
ind2a = np.argmax(Corp_vec)
# Plot the results of the detection
plt.plot(prob_vec,Corp_vec,label='Correct Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec[ind2a])


# In[20]:


ind2b = np.argmin(Mdp_vec)
plt.plot(prob_vec,Fap_vec,label='FA Prob.')
plt.plot(prob_vec,Mdp_vec,label='MD Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec[ind2b])


# In[21]:


prob2 = 0.8
A_sam_thresh2, Corp2, Fap2, Mdp2,KMat2, RMSE2 = noK_prediction_slide(x_pred_sam2, prob2, 2, B2)
print('The probability of detecting K=2 sources is:', Corp2,'% and RMSE =',RMSE2,'for K=2') 
print('The false alarm probability is:',Fap2,'%') 
print('The misdetection probability is:',Mdp2,'%')


# In[22]:


# Optimize the probabilities by chosing the best prob. threshold value 
prob_vec = np.linspace(0.4,1,num=61)
Corp_vec = np.zeros((prob_vec.size,1))
Fap_vec = np.zeros((prob_vec.size,1))
Mdp_vec = np.zeros((prob_vec.size,1))
for n in range(prob_vec.size):
    A_sam_thresh_vec, Corp_vec[n], Fap_vec[n], Mdp_vec[n],KMat, RMSE = noK_prediction_slide(x_pred_sam3, prob_vec[n], 3, GT_angles3)
ind3a = np.argmax(Corp_vec)
# Plot the results of the detection
plt.plot(prob_vec,Corp_vec,label='Correct Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec[ind3a])


# In[22]:


ind3b = np.argmin(Mdp_vec)
plt.plot(prob_vec,Fap_vec,label='FA Prob.')
plt.plot(prob_vec,Mdp_vec,label='MD Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec[ind3b])


# In[23]:


prob3 = 0.8
A_sam_thresh3, Corp3, Fap3, Mdp3,KMat3, RMSE3 = noK_prediction_slide(x_pred_sam3, prob3, 3, B3)
print('The probability of detecting K=3 sources is:', Corp3,'% and RMSE =',RMSE3,'for K=3') 
print('The false alarm probability is:',Fap3,'%') 
print('The misdetection probability is:',Mdp3,'%')


# In[24]:


# The Hausdorff distance 
def Hausdorffdirectdist_1D(A,B):
    h = 0
    for i in range(A.size):
        x=A[i]
        short = math.inf
        for j in range(B.size):
            y=B[j]
            d = np.abs(x-y)
            if d<short:
                short=d
        if short>h:
            h=short
    return h 

def Hausdorffdist_1D(A,B):
    d = max(Hausdorffdirectdist_1D(A,B), Hausdorffdirectdist_1D(B,A))
    return d

def Hausdorffdist_1D_set(A,B):
    dim = A.shape[0]
    dis = 0
    h = 0
    for i in range(dim):
        d = Hausdorffdist_1D(A[i],B[i])
        dis = dis + d
        if d>h:
            h = d
    mean_H = dis/dim        
    return mean_H, h

# Function that removes NaNs in the angles (due to single source - GT)
def remove_nans(y): 
    data_size = y.shape[0]
    y_NOnans =[]
    for n in range(data_size):
        log = np.isnan(y[n])
        val = y[n][np.logical_not(log)]
        y_NOnans.append(val)
    return y_NOnans


# In[25]:


A_sam_thresh1_noNans = np.array(remove_nans(A_sam_thresh1))
A_sam_thresh2_noNans = np.array(remove_nans(A_sam_thresh2))
A_sam_thresh3_noNans = np.array(remove_nans(A_sam_thresh3))


# In[26]:


mean_H1, max_H1 = Hausdorffdist_1D_set(A_sam_thresh1_noNans,B1)
mean_H1 = round(mean_H1,2)
print(mean_H1)
print(max_H1)


# In[27]:


mean_H2, max_H2 = Hausdorffdist_1D_set(A_sam_thresh2_noNans,B2)
mean_H2 = round(mean_H2,2)
print(mean_H2)
print(max_H2)


# In[28]:


mean_H3, max_H3 = Hausdorffdist_1D_set(A_sam_thresh3_noNans,B3)
mean_H3 = round(mean_H3,2)
print(mean_H3)
print(max_H3)


# In[29]:


# Save the results and plot them in MATLAB
filename_s = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/HWU2/Code/Python/DoA Estimation/DoA_Estimation_underdetermined/ComparisonRESULTS/Slide_ang_sep10_K1to3_0dB_T1000_CNN_RQ.h5'
hfs = h5py.File(filename_s, 'w')
hfs.create_dataset('GT_angles1', data=B1)
hfs.create_dataset('GT_angles2', data=B2)
hfs.create_dataset('GT_angles3', data=B3)
hfs.create_dataset('Conf_level1', data=prob1)
hfs.create_dataset('Conf_level2', data=prob2)
hfs.create_dataset('Conf_level3', data=prob3)
hfs.create_dataset('Mean_Hausdorff1', data=mean_H1)
hfs.create_dataset('Mean_Hausdorff2', data=mean_H2)
hfs.create_dataset('Mean_Hausdorff3', data=mean_H3)
hfs.create_dataset('Max_Hausdorff1', data=max_H1)
hfs.create_dataset('Max_Hausdorff2', data=max_H2)
hfs.create_dataset('Max_Hausdorff3', data=max_H3)
hfs.create_dataset('CNN_pred_angles1', data=A_sam_thresh1)
hfs.create_dataset('CNN_pred_angles2', data=A_sam_thresh2)
hfs.create_dataset('CNN_pred_angles3', data=A_sam_thresh3)
hfs.close()


# In[40]:


B1[0]

