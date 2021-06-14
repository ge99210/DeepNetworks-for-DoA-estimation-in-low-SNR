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


# New model trained in the lo-SNR regime - 16/09/2020
# model_CNN = load_model('Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_v1_new_training.h5')
# model_CNN = load_model('Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_v2_10epchdrop.h5')
model_CNN = load_model('Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_Adam_dropRoP_0_7.h5')


# In[3]:


K = 2
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)


# In[4]:


# Load the Test Data for Experiment 1A
filename2 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TEST_DATA_16ULA_K2_0dBSNR_T200_3D_slideang_offgrid_sep2coma11_power_mismatch.h5'
f2 = h5py.File(filename2, 'r')
GT_angles = np.transpose(np.array(f2['angles']))
Ry_sam_test = np.array(f2['sam']) 


# In[5]:


Ry_sam_test.shape


# In[6]:


# First permute the tensor and then predict
[n_test,dim,N,M]=Ry_sam_test.shape
X_test_data_sam = Ry_sam_test.swapaxes(1,3)
X_test_data_sam.shape


# In[7]:


GT_angles.shape


# In[8]:


# The true set
B = GT_angles.T
# Estimation with the covariance matrix
with tf.device('/cpu:0'):
    x_pred_sam = model_CNN.predict(X_test_data_sam)
# Classify K sources with maximum probability - Scenario 1
x_ind_K_sam = np.argpartition(x_pred_sam, -K, axis=1)[:, -K:]
A_sam = np.sort(v[x_ind_K_sam])


# In[9]:


B[10]


# In[10]:


A_sam[10]


# In[11]:


# Calculate the RMSE [in degrees]
RMSE_sam = round(np.sqrt(mean_squared_error(np.sort(A_sam), np.sort(B))),4)
print(RMSE_sam)


# In[12]:


# Store the Data for Experiment 1A
filename3 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/HWU2/Code/Python/DoA Estimation/DoA_Estimation_underdetermined/ComparisonRESULTS/Slide_ang_2coma11sep_K2_0dB_T200_CNN_new_power_mismatch_new.h5'
hf = h5py.File(filename3, 'w')
hf.create_dataset('GT_angles', data=B)
hf.create_dataset('CNN_pred_angles', data=A_sam)
hf.close()


# In[ ]:




