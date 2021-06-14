#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, Softmax, LeakyReLU
from keras.regularizers import l2, l1
from keras.initializers import glorot_normal
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


# In[2]:


# Fix K=1-3 detection of sources - Experiment Part A: 1)
filename1 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TRAIN_DATA_16ULA_K1to3_min15to0dBSNR_res1_3D.h5'
f1 = h5py.File(filename1, 'r')
angles = np.transpose(np.array(f1['angles']))
Ry_the = np.array(f1['theor'])
res = 1


# In[3]:


An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)
print(v)


# In[4]:


DNN_outp = v.size
print(DNN_outp)


# In[5]:


angles.shape


# In[6]:


Ry_the.shape


# In[7]:


[SNRs, n, chan, M, N] = Ry_the.shape
X_data0=Ry_the.swapaxes(2,4)
X_data0.shape


# In[8]:


X_data = X_data0.reshape([SNRs*n,N,M,chan])
X_data.shape


# In[9]:


# this function performs min-max scaling  after fitting to 3D data by reshaping
def min_max_sc_3D(X_data, scaler):
    [tr_samp, N, M, chan] = X_data.shape
    Data = X_data.reshape([tr_samp,M*N*chan])
    scaler.fit(Data)
    X_ = scaler.transform(Data)
    X = X_.reshape([tr_samp, N, M, chan])
    return X


# In[10]:


scaler = MinMaxScaler()
X_data_sc = min_max_sc_3D(X_data, scaler)
X_data_sc.shape


# In[9]:


# Function that removes NaNs in the angles (due to single source - GT)
def remove_nans(y): 
    data_size = y.shape[0]
    y_NOnans =[]
    for n in range(data_size):
        log = np.isnan(y[n])
        val = y[n][np.logical_not(log)]
        y_NOnans.append(val)
    return y_NOnans


# In[10]:


# Create the multiple labels
Y_Labels0 = remove_nans(angles)
mlb = MultiLabelBinarizer()
yTrain_encoded = mlb.fit_transform(Y_Labels0)
yTrain_encoded.shape


# In[11]:


Y_Labels = np.tile(yTrain_encoded, reps=(SNRs,1))
Y_Labels.shape


# In[12]:


# Split the dataset into training and validation sets
xTrain, xVal, yTrain, yVal = train_test_split(X_data, Y_Labels, test_size=0.1, random_state=42) # checked


# In[13]:


# Define the model (CNN) for single source localization
input_shape = xTrain.shape[1:]
kern_size1 = 3
kern_size2 = 2

model = Sequential() # kernel_regularizer=l1(0.00001),
model.add(Conv2D(256, kernel_size=(kern_size1,kern_size1), activation=None, input_shape=input_shape,                 name="Conv2D_1",padding="valid", strides=(2,2)))
model.add(BatchNormalization(trainable=True))
model.add(ReLU())
model.add(Conv2D(256, kernel_size=(kern_size2,kern_size2), activation=None,name="Conv2D_2", padding="valid"))
model.add(BatchNormalization(trainable=True))
model.add(ReLU())
model.add(Conv2D(256, kernel_size=(kern_size2,kern_size2), activation=None,name="Conv2D_3", padding="valid"))
model.add(BatchNormalization(trainable=True))
model.add(ReLU())
model.add(Conv2D(256, kernel_size=(kern_size2,kern_size2), activation=None,name="Conv2D_4", padding="valid"))
model.add(BatchNormalization(trainable=True))
model.add(ReLU())
model.add(Flatten())
model.add(Dense(4096, activation="relu",name="Dense_Layer1"))
model.add(Dropout(0.3,name="Dropout1"))
model.add(Dense(2048, activation="relu",name="Dense_Layer2"))
model.add(Dropout(0.3,name="Dropout2"))
model.add(Dense(1024, activation="relu",name="Dense_Layer3"))
model.add(Dropout(0.3,name="Dropout3"))
model.add(Dense(DNN_outp, activation="sigmoid", kernel_initializer=glorot_normal(seed=None),name="Classif_Layer"))
model.summary()


# In[14]:


# Train the model with Adam and Reduce lr on Plateau
# Last best pars is bs=128, rlr 0.7, pat=10, lr=2e-3 16/4/21 19:00
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, verbose=1)
cbks = [rlr] 
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")])
train_history = model.fit(xTrain, yTrain, epochs=200, batch_size=32, shuffle=True, validation_data=(xVal, yVal), callbacks=cbks)


# In[15]:


model.save('Model_CNN_DoA_N16_K1to3_res1_min15to0dBSNR_Kunknown_adam_bs32_lr1emin3.h5') 


# In[16]:


# summarize history for accuracy
f1 = plt.figure(1)
plt.plot(train_history.history['acc'], label='Training accuracy')
plt.plot(train_history.history['val_acc'], label='Validation accuracy')
#plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='lower right')
plt.grid()
plt.show()

# summarize history for loss
f2 = plt.figure(2)
plt.plot(train_history.history['loss'], label='Training loss')
plt.plot(train_history.history['val_loss'], label='Validation loss')
#plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper left')
plt.grid()
plt.show()


# In[17]:


# Save the figures to include them in the paper
f1.savefig("training_acc_mixK_mixSNR_min15to0dB_bs32_lr1emin3.eps", dpi=1200, bbox_inches='tight')
f2.savefig("training_loss_mixK_mixSNR_min15to0dB_bs32_lr1emin3.eps", dpi=1200, bbox_inches='tight')


# In[18]:


# save the training performance for reporting
f3 = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(train_history.history['acc'], label='Training accuracy')
plt.plot(train_history.history['val_acc'], label='Validation accuracy')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val.'], loc='lower right')
plt.grid()
plt.subplot(2,1,2)
plt.plot(train_history.history['loss'], label='Training loss')
plt.plot(train_history.history['val_loss'], label='Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val.'], loc='upper right')
plt.grid()
plt.show()


# In[19]:


# Save the figures to include them in the paper
f3.savefig("training_perf_mixK_mixSNR_min15to0dB_bs32_lr1emin3.eps", dpi=1200, bbox_inches='tight')


# In[ ]:




