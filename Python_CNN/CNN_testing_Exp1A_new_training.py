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


# New model with low-SNR training - 16/09/2020
model_CNN = load_model('Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_Adam_dropRoP_0_7.h5')


# In[3]:


K = 2
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)


# In[4]:


# Load the Test Data for Experiment 1A
filename2 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7.h5'
f2 = h5py.File(filename2, 'r')
GT_angles = np.transpose(np.array(f2['angles']))
Ry_the_test = np.array(f2['theor'])
Ry_sam_test = np.array(f2['sam']) 


# In[5]:


Ry_sam_test.shape


# In[6]:


# First permute the tensor and then predict
[n_test,dim,N,M]=Ry_sam_test.shape
X_test_data_sam = Ry_sam_test.swapaxes(1,3)
X_test_data_the = Ry_the_test.swapaxes(1,3)
X_test_data_sam.shape


# In[7]:


GT_angles.shape


# In[8]:


# The true set
B = GT_angles.T
# Estimation with the covariance matrix
with tf.device('/cpu:0'):
    # Estimation with the covariance matrix
    x_pred_sam = model_CNN.predict(X_test_data_sam)
# Classify K sources with maximum probability - Scenario 1
x_ind_K_sam = np.argpartition(x_pred_sam, -K, axis=1)[:, -K:]
A_sam = np.sort(v[x_ind_K_sam])


# In[9]:


# Calculate the RMSE [in degrees]
RMSE_sam = round(np.sqrt(mean_squared_error(np.sort(A_sam), np.sort(B))),4)
print(RMSE_sam)


# In[10]:


plt.figure()
plt.scatter(np.arange(B.shape[0]),np.sort(A_sam[:,0])- np.sort(B[:,0]),label='1st Signal')
plt.scatter(np.arange(B.shape[0]),np.sort(A_sam[:,1])- np.sort(B[:,1]),label='1st Signal')
plt.title('CNN errors')
plt.ylim((-1.25, 1.25))
plt.legend()
plt.grid()
plt.show()


# In[11]:


filename3 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/HWU2/Code/Python/DoA Estimation/DoA_Estimation_underdetermined/ComparisonRESULTS/Slide_angsep4coma7_K2_min10dB_T2000_CNN_new_RQ.h5'
hf = h5py.File(filename3, 'w')
hf.create_dataset('GT_angles', data=B)
hf.create_dataset('CNN_pred_angles', data=A_sam)
hf.close()


# In[10]:


# Check here the threshold prediction without assuming knowledge of K, which is here used only for the rate calculation
def noK_prediction(x_pred_sam, prob, K): 
    log = (x_pred_sam > prob).astype(int)
    Dsz = log.shape[0]
    KMat = log.sum(axis=1)
    Kmax = max(KMat)
    Corp = sum(KMat==K)*100/Dsz
    Fap = sum(KMat>K)*100/Dsz
    Mdp = sum(KMat<K)*100/Dsz
    A_sam = np.zeros((Dsz,Kmax))
    x = float("nan")
    for n in range(0,Dsz):
        val = v[log[n]==1]   
        if val.size<Kmax:
            val = np.append(val,np.tile(x,(Kmax-val.size)))  
        A_sam[n] = val
    return A_sam, Corp, Fap, Mdp      


# In[11]:


prob = 0.7
A_sam_thresh, Corp, Fap, Mdp = noK_prediction(x_pred_sam, prob, K)
print('The probability of detecting K signals is:', Corp) 
print('The false alarm probability is:',Fap) 
print('The misdetection probability is:',Mdp)


# In[12]:


# Plot the results of the detection
plt.scatter(np.arange(A_sam_thresh.shape[0]),A_sam_thresh[:,0],label='1st Signal')
plt.scatter(np.arange(A_sam_thresh.shape[0]),A_sam_thresh[:,1],label='2nd Signal')
if A_sam_thresh.shape[1]==3:
    plt.scatter(np.arange(A_sam_thresh.shape[0]),A_sam_thresh[:,2],label='False alarm')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




