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
from sklearn.metrics import mean_squared_error, confusion_matrix
import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras
import seaborn as sn


# In[2]:


# New model trained in the low-SNR regime - 25/09/2020
# model_CNN = load_model('Model_CNN_DoA_N16_K1to3_res1_min15to0dBSNR_Kunknown_bs128_adam_lr0_002.h5')  
# model_CNN = load_model('Model_CNN_DoA_N16_K1to3_res1_min15to0dBSNR_Kunknown_adam_bs64_lr1emin3.h5') # best result so far
model_CNN = load_model('Model_CNN_DoA_N16_K1to3_res1_min15to0dBSNR_Kunknown_adam_bs32_lr1emin3.h5')


# In[3]:


res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)


# In[4]:


# Load the Test Data for Experiment 1A
filename1 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K1_min10dBSNR_T3000_3D_fixedang_offgrid.h5'
f1 = h5py.File(filename1, 'r')
GT_angles1 = np.array(f1['angles'])
Ry_sam_test1 = np.array(f1['sam']) 
Ry_the_test1 = np.array(f1['the']) 
K1 = 1
[n_test,chan,N,M]=Ry_sam_test1.shape
X_test_data_sam1 = Ry_sam_test1.swapaxes(1,3)
X_test_data_the1 = Ry_the_test1.swapaxes(1,3)


# In[5]:


X_test_data_sam1.shape


# In[6]:


# Load the Test Data for Experiment 1A
filename2 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K2_min10dBSNR_T3000_3D_fixedang_offgrid.h5'
f2 = h5py.File(filename2, 'r')
GT_angles2 = np.array(f2['angles'])
Ry_sam_test2 = np.array(f2['sam']) 
Ry_the_test2 = np.array(f2['the']) 
K2 = 2
X_test_data_sam2 = Ry_sam_test2.swapaxes(1,3)
X_test_data_the2 = Ry_the_test2.swapaxes(1,3)


# In[7]:


# Load the Test Data for Experiment 1A
filename3 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K3_min10dBSNR_T3000_3D_fixedang_offgrid.h5'
f3 = h5py.File(filename3, 'r')
GT_angles3 = np.array(f3['angles'])
Ry_sam_test3 = np.array(f3['sam']) 
Ry_the_test3 = np.array(f3['the']) 
K3 = 3
X_test_data_sam3 = Ry_sam_test3.swapaxes(1,3)
X_test_data_the3 = Ry_the_test3.swapaxes(1,3)


# In[8]:


# The true set
B1 = np.tile(GT_angles1,(n_test,1))
B2 = np.tile(GT_angles2,(n_test,1))
B3 = np.tile(GT_angles3,(n_test,1))


# In[9]:


# Estimation with the sample covariance matrix K=1
with tf.device('/cpu:0'):
    x_pred_sam1 = model_CNN.predict(X_test_data_sam1)
x_ind_K_sam1 = np.argpartition(x_pred_sam1, -K1, axis=1)[:, -K1:]
A_sam1 = np.sort(v[x_ind_K_sam1])

# Estimation with the true covariance matrix K=1
with tf.device('/cpu:0'):
    x_pred_the1 = model_CNN.predict(X_test_data_the1)
x_ind_K_the1 = np.argpartition(x_pred_the1, -K1, axis=1)[:, -K1:]
A_the1 = np.sort(v[x_ind_K_the1])


# In[10]:


# Calculate the RMSE [in degrees]
RMSE_sam1 = round(np.sqrt(mean_squared_error(np.sort(A_sam1), np.sort(B1))),4)
# Calculate the RMSE [in degrees]
RMSE_the1 = round(np.sqrt(mean_squared_error(np.sort(A_the1), np.sort(B1))),4)
print('The RMSE with the true cov. mat. is ',RMSE_the1,' and with the sample is ',RMSE_sam1, 'for K=1')


# In[11]:


# Estimation with the sample covariance matrix K=2
with tf.device('/cpu:0'):
    x_pred_sam2 = model_CNN.predict(X_test_data_sam2)
x_ind_K_sam2 = np.argpartition(x_pred_sam2, -K2, axis=1)[:, -K2:]
A_sam2 = np.sort(v[x_ind_K_sam2])

# Estimation with the true covariance matrix K=1
with tf.device('/cpu:0'):
    x_pred_the2 = model_CNN.predict(X_test_data_the2)
x_ind_K_the2 = np.argpartition(x_pred_the2, -K2, axis=1)[:, -K2:]
A_the2 = np.sort(v[x_ind_K_the2])


# In[12]:


# Calculate the RMSE [in degrees]
RMSE_sam2 = round(np.sqrt(mean_squared_error(np.sort(A_sam2), np.sort(B2))),4)
# Calculate the RMSE [in degrees]
RMSE_the2 = round(np.sqrt(mean_squared_error(np.sort(A_the2), np.sort(B2))),4)
print('The RMSE with the true cov. mat. is ',RMSE_the2,' and with the sample is ',RMSE_sam2, 'for K=2')


# In[13]:


# Estimation with the sample covariance matrix K=2
with tf.device('/cpu:0'):
    x_pred_sam3 = model_CNN.predict(X_test_data_sam3)
x_ind_K_sam3 = np.argpartition(x_pred_sam3, -K3, axis=1)[:, -K3:]
A_sam3 = np.sort(v[x_ind_K_sam3])

# Estimation with the true covariance matrix K=1
with tf.device('/cpu:0'):
    x_pred_the3 = model_CNN.predict(X_test_data_the3)
x_ind_K_the3 = np.argpartition(x_pred_the3, -K3, axis=1)[:, -K3:]
A_the3 = np.sort(v[x_ind_K_the3])


# In[14]:


# Calculate the RMSE [in degrees]
RMSE_sam3 = round(np.sqrt(mean_squared_error(np.sort(A_sam3), np.sort(B3))),4)
# Calculate the RMSE [in degrees]
RMSE_the3 = round(np.sqrt(mean_squared_error(np.sort(A_the3), np.sort(B3))),4)
print('The RMSE with the true cov. mat. is ',RMSE_the3,' and with the sample is ',RMSE_sam3, 'for K=3')


# In[15]:


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
    if count==0:
        RMSE = math.inf
    else:
        RMSE = np.sqrt(sum_rmse/K/count)    
    return A_sam, Corp, Fap, Mdp, KMat, RMSE   


# In[16]:


# Optimize the probabilities by chosing the best prob. threshold value 
prob_vec1 = np.linspace(0.7,1,num=31)
Corp_vec1 = np.zeros((prob_vec1.size,1))
Fap_vec1 = np.zeros((prob_vec1.size,1))
Mdp_vec1 = np.zeros((prob_vec1.size,1))
for n in range(prob_vec1.size):
    A_sam_thresh_vec1, Corp_vec1[n], Fap_vec1[n], Mdp_vec1[n],KMat1, RMSE1 = noK_prediction_slide(x_pred_sam1, prob_vec1[n], K1, B1)
ind1a = np.argmax(Corp_vec1)
# Plot the results of the detection
plt.plot(prob_vec1,Corp_vec1,label='Correct Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec1[ind1a])


# In[17]:


ind1b = np.argmin(Fap_vec1)
plt.plot(prob_vec1,Fap_vec1,label='FA Prob.')
plt.plot(prob_vec1,Mdp_vec1,label='MD Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec1[ind1b])


# In[18]:


# Optimize the probabilities by chosing the best prob. threshold value 
prob_vec2 = np.linspace(0.6,1,num=41)
Corp_vec2 = np.zeros((prob_vec2.size,1))
Fap_vec2 = np.zeros((prob_vec2.size,1))
Mdp_vec2 = np.zeros((prob_vec2.size,1))
for n in range(prob_vec2.size):
    A_sam_thresh_vec2, Corp_vec2[n], Fap_vec2[n], Mdp_vec2[n],KMat2, RMSE2 = noK_prediction_slide(x_pred_sam2, prob_vec2[n], K2, B2)
ind2a = np.argmax(Corp_vec2)
# Plot the results of the detection
plt.plot(prob_vec2,Corp_vec2,label='Correct Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec2[ind2a])


# In[19]:


ind2b = np.argmin(Fap_vec2)
plt.plot(prob_vec2,Fap_vec2,label='FA Prob.')
plt.plot(prob_vec2,Mdp_vec2,label='MD Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec2[ind2b])


# In[20]:


# Optimize the probabilities by chosing the best prob. threshold value 
prob_vec3 = np.linspace(0.4,0.8,num=41)
Corp_vec3 = np.zeros((prob_vec3.size,1))
Fap_vec3 = np.zeros((prob_vec3.size,1))
Mdp_vec3 = np.zeros((prob_vec3.size,1))
for n in range(prob_vec3.size):
    A_sam_thresh_vec3, Corp_vec3[n], Fap_vec3[n], Mdp_vec3[n],KMat3, RMSE3 = noK_prediction_slide(x_pred_sam3, prob_vec3[n], K3, B3)
ind3a = np.argmax(Corp_vec3)
# Plot the results of the detection
plt.plot(prob_vec3,Corp_vec3,label='Correct Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec3[ind3a])


# In[21]:


ind3b = np.argmin(Fap_vec3)
plt.plot(prob_vec3,Fap_vec3,label='FA Prob.')
plt.plot(prob_vec3,Mdp_vec3,label='MD Prob.')
plt.legend()
plt.show() 
print('The prob. that maximizes the results is', prob_vec3[ind3b])


# In[18]:


prob1 = 0.9
A_sam_thresh1, Corp1, Fap1, Mdp1,KMat1, RMSE1 = noK_prediction_slide(x_pred_sam1, prob1, K1, B1)
print('The probability of detecting K sources is:', Corp1,'% and RMSE =',RMSE1,'for K=1') 
print('The false alarm probability is:',Fap1,'%') 
print('The misdetection probability is:',Mdp1,'%')


# In[19]:


prob2 = 0.8
A_sam_thresh2, Corp2, Fap2, Mdp2,KMat2, RMSE2 = noK_prediction_slide(x_pred_sam2, prob2, K2, B2)
print('The probability of detecting K sources is:', Corp2,'% and RMSE =',RMSE2,'for K=2') 
print('The false alarm probability is:',Fap2,'%') 
print('The misdetection probability is:',Mdp2,'%')


# In[20]:


prob3 = 0.6
A_sam_thresh3, Corp3, Fap3, Mdp3,KMat3, RMSE3 = noK_prediction_slide(x_pred_sam3, prob3, K3, B3)
print('The probability of detecting K sources is:', Corp3,'% and RMSE =',RMSE3,'for K=3') 
print('The false alarm probability is:',Fap3,'%') 
print('The misdetection probability is:',Mdp3,'%')


# In[48]:


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


# In[59]:


A_sam_thresh1_noNans = np.array(remove_nans(A_sam_thresh1))
A_sam_thresh2_noNans = np.array(remove_nans(A_sam_thresh2))
A_sam_thresh3_noNans = np.array(remove_nans(A_sam_thresh3))


# In[60]:


mean_H1, max_H1 = Hausdorffdist_1D_set(A_sam_thresh1_noNans,B1)
mean_H1 = round(mean_H1,2)
print(mean_H1)
print(max_H1)


# In[61]:


mean_H2, max_H2 = Hausdorffdist_1D_set(A_sam_thresh2_noNans,B2)
mean_H2 = round(mean_H2,2)
print(mean_H2)
print(max_H2)


# In[62]:


mean_H3, max_H3 = Hausdorffdist_1D_set(A_sam_thresh3_noNans,B3)
mean_H3 = round(mean_H3,2)
print(mean_H3)
print(max_H3)


# In[63]:


KMat12 = np.append(KMat1,KMat2)
KMat = np.append(KMat12,KMat3)
GT12 = np.append(K1*np.ones((n_test,1)), K2*np.ones((n_test,1)))
GT = np.append(GT12, K3*np.ones((n_test,1)))


# In[64]:


CM = confusion_matrix(GT,KMat,normalize='true')
print(CM)


# In[67]:


p_conf_lev = [prob1, prob2, prob3]
print(p_conf_lev)


# In[68]:


df_cm = pd.DataFrame(CM*100, range(min(KMat),max(KMat)+1), range(min(KMat),max(KMat)+1))
f = plt.figure(figsize=(7,4))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".2f", cmap='Blues') # font size
plt.xlabel('Predicted K')
plt.ylabel('True K')
plt.show()
f.savefig("Exper6_conf_mat_min10dB_RQ.eps", dpi=1200, bbox_inches='tight')


# In[ ]:




