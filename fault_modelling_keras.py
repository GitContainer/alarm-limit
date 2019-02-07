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
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.layers.core import Activation

from keras import optimizers


###############################################################################
# Set the working directory
###############################################################################
os.chdir('C:\\Users\\40204945\\Documents\\sumitomo')
os.listdir('.')

###############################################################################
# Get and preprocess the data. Preprocessing involves
# 1. keeping only the needed columns
# 2. creating date based index
# 3. determining moving average
###############################################################################
def create_date_index(df):
    df_di = df.rename(columns = {"Unnamed: 0": "datetime"})
    df_di = df_di.sort_values(by = 'datetime')
    df_di = df_di.set_index('datetime')
    return df_di

def column_subsetting(df, col_ind):
    df_s = df.iloc[:,col_ind]
    return df_s

     
def moving_average(df, tw = '5d'):
    ma_df = df.rolling(tw).mean()
    ma_df = ma_df.dropna(axis = 0, how = 'any')
    return ma_df

def preprocess_df(df, col_ind, tw):
    df_di = create_date_index(df)        
    df_s =  column_subsetting(df_di, col_ind)
    ma_df = moving_average(df_s, tw = '5d')
    return ma_df

col_ind = [0, 1, 2, 3, 8]
tw = '5d'
abnormal_df = pd.read_pickle("./abnormal_df.pkl")
ma_df = preprocess_df(abnormal_df, col_ind, tw)

###############################################################################
# 2. plot the preprocessed data
###############################################################################
def plot_df(df):
    for col in df:
        plt.plot(df[col], lw = 0, marker = 'o', ms = 0.03, color = 'b')
        plt.title(col)
        plt.show()
    return
    
plot_df(ma_df)


###############################################################################
# get the reference data for model creation
###############################################################################
def get_reference_data(df, date_range, load_range):
    
    start_date, end_date =  date_range
    ll, hl = load_range
    
    df_ref = df.loc[start_date:end_date]
        
    ind1 = df_ref['FC8215LD.CPV'] > ll
    ind2 = df_ref['FC8215LD.CPV'] < hl
    ind = ind1 & ind2
    
    df_ref = df_ref.loc[ind,:]
    return df_ref

date_range = ['2018-05-01', '2018-11-01']
load_range = [90, 105]
df_ref = get_reference_data(ma_df, date_range, load_range)
plot_df(df_ref)    

###############################################################################
# data (all and reference) normalization before model creation
###############################################################################
scaler = MinMaxScaler()
ma_values = ma_df.values
ma_scaled = scaler.fit_transform(ma_values)

ref_val = df_ref.values
ref_scaled = scaler.transform(ref_val)

###############################################################################
# create a model for the reference data
###############################################################################
def get_operation_model(ref_data):
    
    x = ref_data[:,0]
    x = x.reshape(-1,1)
    y = ref_data[:,1:]
    
    _, n_input = x.shape
    _, n_output = y.shape
       
    OPTIMIZER = Adam()
    NB_EPOCH = 15
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
    
    return model


op_model = get_operation_model(ref_scaled)

###############################################################################
# prediction based on the operation model
###############################################################################
x_ref = ref_scaled[:,0]
x_ref = x_ref.reshape(-1,1)
y_ref = ref_scaled[:,1:]

y_ref_predicted = op_model.predict(x_ref)

_, n = y_ref_predicted.shape
for i in range(n):
    plt.plot(y_ref[:,i], label = 'Observed')
    plt.plot(y_ref_predicted[:,i], label = 'Predicted')
    plt.legend()
    plt.show()

###############################################################################
# Autoencoder model for reducing the dimensionality
###############################################################################
def get_autoencoder_model(ref_scaled, encoding_dim):
    
    y = ref_scaled[:,1:]
    _, n = y.shape
    
    input_val = Input(shape=(n,))
    encoded = Dense(3, activation='sigmoid')(input_val)
    encoded = Dense(2, activation='sigmoid')(encoded)
    
    encoded = Dense(1, activation='sigmoid')(encoded)
    
    decoded = Dense(2, activation='sigmoid')(encoded)
    decoded = Dense(3, activation='sigmoid')(decoded)
    decoded = Dense(n, activation = 'sigmoid')(decoded)
    
    # autoencoder
    autoencoder = Model(input_val, decoded)
    
    # encoder
    encoder = Model(input_val, encoded)
        
    autoencoder.compile(optimizer='adam', loss='mse')
    
    autoencoder.fit(y, y,
                    epochs=15,
                    shuffle=True)
    
    return autoencoder, encoder


autoencoder, encoder = get_autoencoder_model(ref_scaled, 1)

###############################################################################
# checking encoder, decoder prediction
###############################################################################
def get_abnormality_index(ma_scaled, op_model, encoder):
    
    x = ma_scaled[:,0]
    x = x.reshape(-1,1)
    y = ma_scaled[:,1:]
    
    ypred = op_model.predict(x)
    
    encoded_y = encoder.predict(y)
    encoded_ypred = encoder.predict(ypred)
        
    dist = np.sqrt(np.square(encoded_y - encoded_ypred))
    return dist

ab_index = get_abnormality_index(ma_scaled, op_model, encoder)

###############################################################################
# Plotting abnormality index
###############################################################################
def plot_abnormality_index(ma_df, ab_index):       
    f, ax = plt.subplots() 
    ax.plot(ma_df.iloc[:,0], lw = 0, color = 'blue', marker = 'o', ms = 0.03, label = 'load')
    ax.set_ylabel(ma_df.columns[0], color='blue')
    ax.tick_params(axis='y', colors='blue')
    ax0 = ax.twinx()
    ax0.plot(ma_df.index.values, ab_index, color = 'red', lw = 0, marker = 'o', ms = 0.03,
             label = 'Deviation')
    ax0.set_ylabel('Deviation From Normal Operation', color='red')
    ax0.tick_params(axis='y', colors='red')
    ax0.spines['right'].set_color('red')
    ax0.spines['left'].set_color('blue')
    return ax

ax = plot_abnormality_index(ma_df, ab_index)
