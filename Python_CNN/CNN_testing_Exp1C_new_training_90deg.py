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

model_CNN = load_model('Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_90deg_v6c.h5') 


# In[3]:


K = 2
res = 1
An_max = 90
An_min = -90
v = np.arange(An_min, An_max+res,res)


# In[4]:


# Load the Test Data for Experiment 1A
filename2 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K2_min10dBSNR_T2000_3D_slideang_offgrid_sep4coma7_90deg.h5'
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


B.shape


# In[10]:


# Calculate the RMSE [in degrees]
RMSE_sam = round(np.sqrt(mean_squared_error(np.sort(A_sam), np.sort(B))),4)
print(RMSE_sam)


# In[11]:


# Plot the results of the detection
plt.scatter(np.arange(A_sam.shape[0]),A_sam[:,0]-B[:,0],label='1st Signal')
plt.scatter(np.arange(A_sam.shape[0]),A_sam[:,1]-B[:,1],label='2nd Signal')
plt.legend()
plt.grid()
plt.show()


# In[12]:


# Plot the results of the detection
plt.scatter(np.arange(A_sam.shape[0]),B[:,0],label='1st Signal', marker='.')
plt.scatter(np.arange(A_sam.shape[0]),B[:,1],label='2nd Signal', marker='.')
plt.scatter(np.arange(A_sam.shape[0]),A_sam[:,0],label='est. 1st Signal')
plt.scatter(np.arange(A_sam.shape[0]),A_sam[:,1],label='est. 2nd Signal')
plt.legend()
plt.grid()
plt.show()


# In[14]:


filename3 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/HWU2/Code/Python/DoA Estimation/DoA_Estimation_underdetermined/ComparisonRESULTS/Slide_angsep4coma7_K2_min10dB_T2000_CNN_new_RQ_90deg_vf6c.h5'
hf = h5py.File(filename3, 'w')
hf.create_dataset('GT_angles', data=B)
hf.create_dataset('CNN_pred_angles', data=A_sam)
hf.close()




