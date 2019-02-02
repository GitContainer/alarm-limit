# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:39:41 2019

@author: Ravi Tiwari
"""

from keras.layers import Input, Dense
from keras.models import Model
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.layers.core import Activation

from keras.callbacks import TensorBoard
from keras import optimizers


###############################################################################
# Set the working directory
###############################################################################
os.chdir('C:\\Users\\40204945\\Documents\\sumitomo')
os.listdir('.')


###############################################################################
# Read and get the moving average
###############################################################################
abnormal_df = pd.read_pickle("./abnormal_df.pkl")

# 1. get the subset and create datetime index
col_ind = [0, 1,2,3,4,10]
df_s = abnormal_df.iloc[:,col_ind]
df_s = df_s.rename(columns = {"Unnamed: 0": "datetime"})
df_s = df_s.sort_values(by = 'datetime')
df_s = df_s.set_index('datetime')

# 2. get the moving average of the whole data set for training scaler
tp = '5d'
ma_df = df_s.rolling('5d').mean()
ma_df = ma_df.dropna(axis = 0, how = 'any')

for i in range(5):
    ma_df.plot(y = i, lw = 0, marker = 'o', ms = 0.03)

scaling_data = ma_df.values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(scaling_data)

for i in range(5):
    plt.plot(scaled_data[:,i])
    plt.show()

###############################################################################
# 2. subsetting the data for creating the model
###############################################################################
start_date = '2018-05-01'
end_date = '2018-11-01'
df_ds = df_s.loc[start_date:end_date]

# 3. subsetting based on load
ind1 = df_ds['FC8215LD.CPV'] > 90
ind2 = df_ds['FC8215LD.CPV'] < 105
ind = ind1 & ind2

# 4. get the subsetted dataframe
df_ds = df_ds.loc[ind,:]


# 5. get the moving average
tp = '5d'
ma_sdf = df_ds.rolling('5d').mean()
ma_sdf = ma_sdf.dropna(axis = 0, how = 'any')

# 6. check the value by plotting
for i in range(5):
    ma_sdf.plot(y = i, lw = 0, marker = 'o', ms = 0.03)


###############################################################################
# make a model to get all the values from the load
###############################################################################    
model_data = ma_sdf.values
scaled_md = scaler.transform(model_data)

# checked the scaled model data
for i in range(5):
    plt.plot(scaled_md[:,i])
    plt.show()

x = scaled_md[:,0]
x = x.reshape(-1,1)
y = scaled_md[:,1:]

_, n_input = x.shape
_, n_output = y.shape

OPTIMIZER = Adam()
NB_EPOCH = 20
VALIDATION_SPLIT=0.1


model = Sequential()
model.add(Dense(n_output, input_shape=(n_input,), kernel_initializer="glorot_uniform"))
model.add(Activation('sigmoid'))
model.add(Dense(n_output, kernel_initializer="glorot_uniform"))
model.add(Activation('sigmoid'))
model.add(Dense(n_output, kernel_initializer="glorot_uniform"))
model.summary()
model.compile(loss='mse', optimizer=OPTIMIZER, metrics=['mse'])

model.fit(x, y, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT)

###############################################################################
# prediction with the trained model
###############################################################################
y_predict = model.predict(x)

###############################################################################
# checking prediction quality using the actual values
###############################################################################
for i in range(4): 
    plt.plot(y[:,i])
    plt.plot(y_predict[:,i])    
    plt.show()

###############################################################################
# Check model prediction on all other data 
###############################################################################
scaled_data
x_all = scaled_data[:,0]
x_all = x_all.reshape(-1,1)
y_all = scaled_data[:,1:]

y_all_predict = model.predict(x_all)

for i in range(4): 
    plt.plot(y_all[:,i])
    plt.plot(y_all_predict[:,i])    
    plt.show()












###############################################################################
# Next Autoencoder (working)
###############################################################################
input_data = ma_sdf.iloc[:,[1,2,3,4]].values
_, n = input_data.shape
#input_data = ma_sdf.values

# check the input data
#plt.plot(input_data[:,0])
#plt.plot(input_data[:,1])

# Simple autoencoder
encoding_dim = 1

input_val = Input(shape=(n,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='sigmoid')(input_val)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(n, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_val, decoded)
encoder = Model(input_val, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')

# prepare the input data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(input_data)

autoencoder.fit(x_train, x_train,
                epochs=20,
                shuffle=True)


encoded_val = encoder.predict(x_train)
decoded_val = decoder.predict(encoded_val)

plt.plot(encoded_val)

for i in range(n):
    plt.plot(x_train[:,i])
    plt.plot(decoded_val[:,i])
    plt.show()

# whenever there are multiple encoded values then can check how they are doing
np.corrcoef(encoded_val[:,0], encoded_val[:,1])


###############################################################################
# deep autoencoder
###############################################################################
input_data = ma_sdf.iloc[:,[1,2,3,4]].values
_, n = input_data.shape
#input_data = ma_sdf.values


# Simple autoencoder
encoding_dim = 1

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

input_val = Input(shape=(n,))
encoded = Dense(3, activation='sigmoid')(input_val)
encoded = Dense(2, activation='sigmoid')(encoded)

encoded = Dense(1, activation='sigmoid')(encoded)

decoded = Dense(2, activation='sigmoid')(encoded)
decoded = Dense(3, activation='sigmoid')(decoded)
decoded = Dense(n, activation='sigmoid')(decoded)

# autoencoder
autoencoder = Model(input_val, decoded)

# encoder
encoder = Model(input_val, encoded)

# decoder
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
#decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adam', loss='mse')

# prepare the input data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(input_data)

autoencoder.fit(x_train, x_train,
                epochs=20,
                shuffle=True)



# get the encoded and decoded values
decoded_val = autoencoder.predict(x_train)
for i in range(4):
    plt.plot(x_train[:,i], label = 'original')
    plt.plot(decoded_val[:,i], label = 'decoded')
    plt.legend()
    plt.show()
    

encoded_train = encoder.predict(x_train)


###############################################################################
# deep autoencoder (result on other data)
###############################################################################
df_s.head()

tp = '5d'
ma = df_s.rolling('5d').mean()
ma = ma.dropna(axis = 0, how = 'any')

# 6. check the value by plotting
for i in range(5):
    ma.plot(y = i, lw = 0, marker = 'o', ms = 0.03)

input_data = ma.iloc[:,[1,2,3,4]].values
x = scaler.transform(input_data)

encoded_test = encoder.predict(x)

plt.plot(encoded_train)
plt.plot(encoded_test)


