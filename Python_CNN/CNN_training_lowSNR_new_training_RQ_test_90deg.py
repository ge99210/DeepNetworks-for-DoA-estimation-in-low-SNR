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
#from tensorflow.keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


# In[2]:


# Fix K detection of sources - Experiment Part A: 1)
filename1 = 'C:/Users/geo_p/OneDrive - Heriot-Watt University/DoA DATA/DoA_DATA_JOURNALS/TRAIN_DATA_16ULA_K2_low_SNR_res1_3D_90deg.h5'
f1 = h5py.File(filename1, 'r')
angles = np.transpose(np.array(f1['angles']))
Ry_the = np.array(f1['theor'])
res = 1
K=2


# In[3]:


An_max = np.max(angles)
An_min = np.min(angles)
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


# In[8]:


X_data0=Ry_the.swapaxes(2,4)
X_data0.shape


# In[9]:


X_data = X_data0.reshape([SNRs*n,N,M,chan])
X_data.shape


# In[10]:


mlb = MultiLabelBinarizer()
yTrain_encoded = mlb.fit_transform(angles)


# In[11]:


yTrain_encoded[1]


# In[12]:


Y_Labels = np.tile(yTrain_encoded, reps=(SNRs,1))
#Y_Labels = yTrain_encoded
Y_Labels.shape


# In[13]:


# Split the dataset into training and validation sets
xTrain, xVal, yTrain, yVal = train_test_split(X_data, Y_Labels, test_size=0.1, random_state=42) # checked


# In[16]:


# Define the model (CNN) for single source localization
input_shape = xTrain.shape[1:]
kern_size1 = 3
kern_size2 = 2

model = Sequential() 
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


# In[17]:


# Train the model with Adam
# Train the model with decaying learn rate
# OPTION 1
def schedule(epoch,lr): # use this function to gradually reduce the lr
    if epoch<1:
        return lr
    else:
        return float(lr*tf.math.exp(-0.1))
# OPTION 2
def step_decay(epoch, lr): # or use this function to reduce every epochs_drop by a desired factor
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 20
    lrate = initial_lr* math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
#dlr = LearningRateScheduler(step_decay,verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, verbose=1)
cbks = [rlr]
#opt = tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9,nesterov=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt , loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")])
train_history = model.fit(xTrain, yTrain, epochs=200, batch_size=32, shuffle=True, validation_data=(xVal, yVal), callbacks=cbks)


# In[18]:


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


# In[19]:


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


# In[20]:


f1.savefig("binary_acc_Adam_LRdef_90deg_v6c.eps", dpi=1200, bbox_inches='tight')
f2.savefig("loss_Adam_LRdef_90deg_v6c.eps", dpi=1200, bbox_inches='tight')


# In[21]:


# Save the figures to include them in the paper
f3.savefig("training_perf_Adam_LRdef_90deg_v6c.eps", dpi=1200, bbox_inches='tight')


# In[22]:


model.save('Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_90deg_v6c.h5') 


# In[ ]:




