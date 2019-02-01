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

from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam

from keras.callbacks import TensorBoard
from time import time
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

# 2. subsetting based on date index
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
input_data =  ma_sdf.iloc[:,0].values 
input_data = input_data.reshape(-1,1)
output_data =  ma_sdf.iloc[:,1:].values

n_input = 1
n_output = 4

OPTIMIZER = Adam()
NB_EPOCH = 50
VALIDATION_SPLIT=0.1


model = Sequential()
model.add(Dense(n_output, input_shape=(n_input,), kernel_initializer="glorot_uniform"))
model.add(Activation('sigmoid'))
model.add(Dense(n_output, kernel_initializer="glorot_uniform"))
model.add(Activation('sigmoid'))
model.add(Dense(n_output, kernel_initializer="glorot_uniform"))
model.summary()
model.compile(loss='mse', optimizer=OPTIMIZER, metrics=['mse'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(input_data, output_data, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, 
                    verbose=1, callbacks=[tensorboard])

###############################################################################
# prediction with the trained model
###############################################################################
y_predict = model.predict(input_data)

###############################################################################
# checking prediction quality using the actual values
###############################################################################
for i in range(4):    
    plt.plot(y_predict[:,i])
    plt.plot(output_data[:,i])
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
    

encoded_value = encoder.predict(x_train)






