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
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.layers.core import Activation

from keras import optimizers


###############################################################################
# Set the working directory
###############################################################################
#1. Laptop
os.chdir('C:\\Users\\40204945\\Documents\\sumitomo')
os.listdir('.')

# 2.workstation
os.chdir('D:\\sumitomo\\code\\alarm-limit')
os.listdir('.')

###############################################################################
# Get and preprocess the data. Preprocessing involves
# 1. keeping only the needed columns
# 2. creating date based index
# 3. determining moving average
###############################################################################
def create_date_index(df):
    df_di = df.rename(columns = {"Unnamed: 5": "datetime"})
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
#    df_di = create_date_index(df)  
    df_sr = df.sort_values(by = 'datetime')  
    df_s =  column_subsetting(df_sr, col_ind)
    ma_df = moving_average(df_s, tw = '5d')
    return ma_df

###############################################################################
# Load and preprocess the data
###############################################################################
#abnormal_df = pd.read_pickle("./df_8220_pg.pkl")

nr_12 = pd.read_pickle("./df_1220_nr.pkl")
abnr_12 = pd.read_pickle("./df_1220_pg.pkl")

nr_82 = pd.read_pickle("./df_8220_tm.pkl")
abnr_82 = pd.read_pickle("./df_8220_pg.pkl")



normal_12_df = pd.read_pickle("./df_1220_nr.pkl")
abnormal_12_df = pd.read_pickle("./df_1220_pg.pkl")

nr_col = [9, 6, 7, 1, 11, 12]
abnr_col = [0, 1, 2, 3, 8, 9]
tw = '1d'

# col names
nr_12.columns[nr_col]
abnr_12.columns[abnr_col]

nr_82.columns[nr_col]
abnr_82.columns[abnr_col]

# moving average for both the reboilers

nr_12_ma = preprocess_df(nr_12, nr_col, tw)
abnr_12_ma = preprocess_df(abnr_12, abnr_col, tw)

nr_82_ma = preprocess_df(nr_82, nr_col, tw)
abnr_82_ma = preprocess_df(abnr_82, abnr_col, tw)

###############################################################################
# 2. plot the preprocessed data
###############################################################################
def plot_df(df):
    for col in df:
        plt.plot(df[col], lw = 0, marker = 'o', ms = 0.03, color = 'b')
        plt.title(col)
        plt.xticks(rotation='vertical')
        plt.show()
    return
    
#plot_df(ma_df)
plot_df(nr_12_ma)
plot_df(nr_82_ma)
plot_df(abnr_82_ma)
plot_df(abnr_12_ma)

plot_df(abnr_12)
plot_df(nr_12)

plot_df(abnr_82)

# individual plot
_, n = abnr_12.shape

# which columns do I want
for i, col in enumerate(abnr_12.columns):
    string = '{:15s}{}'.format(col, i)
    print(string)

start_date = '2017-01'
end_date = '2017-03'
col_names = ['FC1215LD.CPV', 'FC1228c.PV', 'FI1228A.PV']

for col in col_names:
    x = abnr_12.loc[start_date : end_date, col]
    x.plot(lw = 0, marker = 'o', ms = 0.03, color = 'b')
    plt.title(col)
    plt.show()



abnr_12.loc['2017-01':'2017-03', 'FI1228A.PV']


for i in range(n):
    avg = abnr_12.iloc[:,i].mean()
    std = abnr_12.iloc[:,i].std()
    abnr_12.plot(y = i, lw = 0, marker = 'o', ms = 0.03, color = 'b')
    plt.ylim([avg - 2*std, avg + 2*std])
    plt.xlim(x_lim)
    plt.show()

abnr_12.plot(y = 3, lw = 0, marker = 'o', ms = 0.03, color = 'b')
plt.ylim([115,125])


###############################################################################
# Bokeh plotting
###############################################################################
from bokeh.plotting import figure, output_file, show

start_date = '2017-01'
end_date = '2017-03'
col_names = ['FC1215LD.CPV', 'FC1228c.PV', 'FI1228A.PV']
col = 'FI1228A.PV'
dat = abnr_12.loc[start_date:end_date, col]
x = dat.index
y = dat.values

# prepare some data
#x = [1, 2, 3, 4, 5]
#y = [6, 7, 2, 4, 5]

# output to static HTML file
output_file("lines.html", title="line plot example")

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(x, y, legend="Temp.", line_width=2)

# show the results
show(p)


###############################################################################
# multiple plot in bokeh
###############################################################################
from bokeh.layouts import column

# prepare some data
start_date = '2017-01'
end_date = '2017-03'
col_names = ['FC1215LD.CPV', 'FC1228c.PV', 'FI1228A.PV']
col = 'FI1228A.PV'
dat = abnr_12.loc[start_date:end_date, col_names]
x = pd.to_datetime(dat.index)
y0 = dat.values[:,0]
y1 = dat.values[:,1]
y2 = dat.values[:,2]


# create a new plot
s1 = figure(width=1500, plot_height=250, title=col_names[0], x_axis_type="datetime")
s1.line(x, y0, color="navy", line_width=2)

# NEW: create a new plot and share both ranges
s2 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[1], x_axis_type="datetime")
s2.line(x, y1, color="firebrick", alpha=0.5)

# NEW: create a new plot and share only one range
s3 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[2], x_axis_type="datetime")
s3.line(x, y2, color="olive", alpha=0.5)

# show the result
show(column(s1, s2, s3))


###############################################################################
# multiple plot 3
###############################################################################
start_date = '2017-01'
end_date = '2018-01'
col_names = ['FI1228A.PV', 'PI1228.PV', 'TI1221A.PV', 'PI1221.PV']
dat = abnr_12.loc[:, col_names]
dat = dat.sort_values(by = 'datetime')
x = pd.to_datetime(dat.index)
y0 = dat.values[:,0]
y1 = dat.values[:,1]
y2 = dat.values[:,2]
y3 = dat.values[:,3]

# create a new plot
s1 = figure(width=1500, plot_height=250, title=col_names[0], x_axis_type="datetime")
s1.line(x, y0, color="navy", line_width=2)

# NEW: create a new plot and share both ranges
s2 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[1], x_axis_type="datetime")
s2.line(x, y1, color="firebrick", alpha=0.5)

# NEW: create a new plot and share only one range
s3 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[2], x_axis_type="datetime")
s3.line(x, y2, color="olive", alpha=0.5)

s4 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[3], x_axis_type="datetime")
s4.line(x, y3, color="olive", alpha=0.5)


# show the result
show(column(s1, s2, s3, s4))


###############################################################################
# multiple plot 4
###############################################################################
col_names = ['FC1215LD.CPV','FI1228A.PV', 'PI1228.PV', 'TI1221A.PV', 'PI1221.PV']
dat = abnr_12.loc[:, col_names]
dat = dat.sort_values(by = 'datetime')
x = pd.to_datetime(dat.index)
y0 = dat.values[:,0]
y1 = dat.values[:,1]
y2 = dat.values[:,2]
y3 = dat.values[:,3]
y4 = dat.values[:,4]

# create a new plot
s1 = figure(width=1500, plot_height=250, title=col_names[0], x_axis_type="datetime")
s1.line(x, y0, color="navy", line_width=2)

# NEW: create a new plot and share both ranges
s2 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[1], x_axis_type="datetime")
s2.line(x, y1, color="firebrick", alpha=0.5)

# NEW: create a new plot and share only one range
s3 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[2], x_axis_type="datetime")
s3.line(x, y2, color="olive", alpha=0.5)

s4 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[3], x_axis_type="datetime")
s4.line(x, y3, color="olive", alpha=0.5)

s5 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[4], x_axis_type="datetime")
s5.line(x, y4, color="olive", alpha=0.5)

# show the result
show(column(s1, s2, s3, s4, s5))


###############################################################################
# multiple moving average plot
###############################################################################
col_names = ['FC1215LD.CPV','FI1228A.PV', 'PI1228.PV', 'TI1221A.PV', 'PI1221.PV']
dat = abnr_12.loc[:, col_names]
dat = dat.sort_values(by = 'datetime')
tw = '5d'
ma_dat = dat.rolling(tw).mean()

###############################################################################
# time series data
###############################################################################
x = pd.to_datetime(dat.index)
y0 = dat.values[:,0]
y1 = dat.values[:,1]
y2 = dat.values[:,2]
y3 = dat.values[:,3]
y4 = dat.values[:,4]

###############################################################################
# moving averge data
###############################################################################
mx = pd.to_datetime(ma_dat.index)
my0 = ma_dat.values[:,0]
my1 = ma_dat.values[:,1]
my2 = ma_dat.values[:,2]
my3 = ma_dat.values[:,3]
my4 = ma_dat.values[:,4]


# create a new plot
s1 = figure(width=1500, plot_height=250, title=col_names[0], x_axis_type="datetime")
#s1.line(x, y0, color="blue", alpha=0.5)
s1.line(mx, my0, color = 'red', line_width = 2)

#show(s1)

# NEW: create a new plot and share both ranges
s2 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[1], x_axis_type="datetime")
#s2.line(x, y1, color="blue", alpha=0.5)
s2.line(mx, my1, color = 'red', line_width = 2)
#show(column(s1, s2))

# NEW: create a new plot and share only one range
s3 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[2], x_axis_type="datetime")
#s3.line(x, y2, color="blue", alpha=0.5)
s3.line(mx, my2, color="red", line_width = 2)
#show(column(s1, s2, s3))

s4 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[3], x_axis_type="datetime")
#s4.line(x, y3, color="blue", alpha=0.5)
s4.line(mx, my3, color="red", line_width = 2)
#show(column(s1, s2, s3, s4))


#s1 = figure(width=1500, plot_height=250, title=col_names[0], x_axis_type="datetime")
#s1.line(x, y0, color="blue", line_width=2)
#s1.line(mx, my0, color = 'red', line_width = 2)

s5 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[4], x_axis_type="datetime")
#s5.line(x, y4, color="blue", alpha=0.5)
s5.line(mx, my4, color="red", line_width = 2)

#show(column(s1, s5))
# show the result
show(column(s1, s2, s3, s4, s5))



###############################################################################
# multiple plot 5 with moving average
###############################################################################
col_names = ['FC1215LD.CPV','FI1228A.PV', 'PI1228.PV', 'TI1221A.PV', 'PI1221.PV']
dat = abnr_12.loc[:, col_names]
dat = dat.sort_values(by = 'datetime')
tw = '5d'
ma_dat = dat.rolling(tw).mean()

###############################################################################
# time series data
###############################################################################
x = pd.to_datetime(dat.index)
y0 = dat.values[:,0]
y1 = dat.values[:,1]
y2 = dat.values[:,2]
y3 = dat.values[:,3]
y4 = dat.values[:,4]

###############################################################################
# moving averge data
###############################################################################
mx = pd.to_datetime(ma_dat.index)
my0 = ma_dat.values[:,0]
my1 = ma_dat.values[:,1]
my2 = ma_dat.values[:,2]
my3 = ma_dat.values[:,3]
my4 = ma_dat.values[:,4]


# create a new plot
s1 = figure(width=1500, plot_height=250, title=col_names[0], x_axis_type="datetime")
s1.line(x, y0, color="blue", alpha=0.5)
s1.line(mx, my0, color = 'red', line_width = 2)

#show(s1)

# NEW: create a new plot and share both ranges
s2 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[1], x_axis_type="datetime")
s2.line(x, y1, color="blue", alpha=0.5)
s2.line(mx, my1, color = 'red', line_width = 2)
#show(column(s1, s2))

# NEW: create a new plot and share only one range
s3 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[2], x_axis_type="datetime")
s3.line(x, y2, color="blue", alpha=0.5)
s3.line(mx, my2, color="red", line_width = 2)
#show(column(s1, s2, s3))

s4 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[3], x_axis_type="datetime")
s4.line(x, y3, color="blue", alpha=0.5)
s4.line(mx, my3, color="red", line_width = 2)
#show(column(s1, s2, s3, s4))


#s1 = figure(width=1500, plot_height=250, title=col_names[0], x_axis_type="datetime")
#s1.line(x, y0, color="blue", line_width=2)
#s1.line(mx, my0, color = 'red', line_width = 2)

s5 = figure(width=1500, height=250, x_range=s1.x_range, title=col_names[4], x_axis_type="datetime")
s5.line(x, y4, color="blue", alpha=0.5)
s5.line(mx, my4, color="red", line_width = 2)

#show(column(s1, s5))
# show the result
show(column(s1, s2, s3, s4, s5))



###############################################################################
# Capturig bokeh output
###############################################################################
from bokeh.models.widgets import Select
from bokeh.io import output_notebook, show, vform
from bokeh.models import CustomJS

output_notebook()
states=['VA','MD','SC']

select = Select(title="Select State:", value="VA", options=states)

show(vform(select))

def call(attr,old,new):
    print(new)
    
select.on_change('value', call)
























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
scaler = StandardScaler()
ma_values = ma_df.values
ma_scaled = scaler.fit_transform(ma_values)

ref_val = df_ref.values
ref_scaled = scaler.transform(ref_val)

###############################################################################
# plot scaled data
###############################################################################
for i in range(1,5):
    plt.scatter(ma_df.index, ma_scaled[:,i], label = ma_df.columns[i])
    plt.ylim([-1.5,1.5])
    plt.legend()
plt.show()

    







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
