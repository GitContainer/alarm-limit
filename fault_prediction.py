# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:11:16 2019

@author: Ravi Tiwari
"""

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import sys

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy.stats as stats
import seaborn as sns
from sys import getsizeof

###############################################################################
# Set the working directory
###############################################################################
os.chdir('C:\\Users\\40204945\\Documents\\sumitomo')
os.listdir('.')

###############################################################################
# Step 1: Read the pickled data
###############################################################################
abnormal_df = pd.read_pickle("./abnormal_df.pkl")
plug_df = pd.read_pickle("./plug_df.pkl")

sorted_df = abnormal_df.sort_values(by = 'Unnamed: 0')
sorted_df = plug_df.sort_values(by = 'Unnamed: 0')

###############################################################################
# Just plotting the sorted df
###############################################################################
scale_dic = {}
scale_dic['abnormal']= {}
scale_dic['abnormal']['FC8215LD.CPV'] = [60, 115]
scale_dic['abnormal']['TI8221A.PV'] = [116, 122]
scale_dic['abnormal']['PI8221.PV'] = [180, 260]
scale_dic['abnormal']['FC8228D.PV'] = [8, 11]
scale_dic['abnormal']['FI8228A.PV'] = [2.5, 4]
scale_dic['abnormal']['PI8228.PV'] = [160, 260]
scale_dic['abnormal']['LC8220.PV'] = [66,73]
scale_dic['abnormal']['PC8223A.PV'] = [104, 106]
scale_dic['abnormal']['PI8223B.PV'] = [106, 108]
scale_dic['abnormal']['PI8223B.PV'] = [38.50, 39.50]
scale_dic['plug']= {}
scale_dic['plug']['FC1215LD.CPV'] = [60, 115]
scale_dic['plug']['TI1221A.PV'] = [115, 129]
scale_dic['plug']['PI1221.PV'] = [160, 260]
scale_dic['plug']['FC1228c.PV'] = [3.5, 5.5]
scale_dic['plug']['FI1228A.PV'] = [1.2, 3.2]
scale_dic['plug']['PI1228.PV'] = [150, 280]
scale_dic['plug']['LC1220.PV'] = [55, 85]
scale_dic['plug']['PC1223A.PV'] = [100, 110]
scale_dic['plug']['PI1223B.PV'] = [100, 110]
scale_dic['plug']['PI1223B.PV'] = [21, 25]

def plot_ts(i, df, dname, scale_dic, ax):
    colname = df.columns[i]
    y_min, y_max = scale_dic[dname][colname]
    x = df.iloc[:,0]    
    y = df.iloc[:,i]
    ax.plot(x,y, lw = 0, marker = 'o', ms = 0.03)
    ax.set_ylim(y_min, y_max)
    ax.set_title(colname)
    ax.tick_params(axis='x', rotation=90)
    return ax

# valid column numbers: 1, 2, 3, 4, 9, 10 
col_nos = [1, 2, 3, 4, 9, 10]
for i in col_nos:
    fig, ax = plt.subplots()
    ax = plot_ts(i, abnormal_df, 'abnormal', scale_dic, ax)
    plt.show()


col_nos = [1, 2, 3, 4, 9, 10]
for i in col_nos:
    fig, ax = plt.subplots()
    ax = plot_ts(i, plug_df, 'plug', scale_dic, ax)
    plt.show()
    

###############################################################################
# get normalized data
###############################################################################
def get_scaled_df(df):
    sorted_df = df.sort_values(by = 'Unnamed: 0')
    scaler = StandardScaler()
    ti_df = sorted_df.set_index('Unnamed: 0')
    np_scaled = scaler.fit_transform(ti_df)
    scaled_df = pd.DataFrame(np_scaled, columns = ti_df.columns, index = ti_df.index)
    return scaled_df
        
abnormal_sc = get_scaled_df(abnormal_df)
plug_sc = get_scaled_df(plug_df)


###############################################################################
# plot normalized data
###############################################################################
def plot_scaled_ts(i, df, ax):
    colname = df.columns[i]
    y_min, y_max = -2, 2
    ax.plot(df.iloc[:,i], lw = 0, marker = 'o', ms = 0.03)
    ax.set_ylim(y_min, y_max)
    ax.set_title(colname)
    ax.tick_params(axis='x', rotation=90)
    return ax

col_nos = [0, 1, 2, 3, 8, 9]
for i in col_nos:    
    fig, ax = plt.subplots()
    ax = plot_scaled_ts(i, abnormal_sc, ax)
    plt.show()

col_nos = [0, 1, 2, 3, 8, 9]
for i in col_nos:    
    fig, ax = plt.subplots()
    ax = plot_scaled_ts(i, plug_sc, ax)
    plt.show()

###############################################################################
# plot moving average on the scaled data
###############################################################################
def plot_ma_on_ts(i, df, tp, ax):
    colname = df.columns[i]
    ma_df = df.rolling(tp).mean()    
    y_min, y_max = -2, 2
    ax.plot(df.iloc[:,i], lw = 0, marker = 'o', ms = 0.03)
    ax.plot(ma_df.iloc[:,i], lw = 0, marker = 'o', ms = 0.03)
    ax.set_ylim(y_min, y_max)
    ax.set_title(colname)
    ax.tick_params(axis='x', rotation=90)
    return ax

tp = '10d'
col_nos = [0, 1, 2, 3, 8, 9]
for i in col_nos:    
    fig, ax = plt.subplots()
    ax = plot_ma_on_ts(i, abnormal_sc, tp, ax)
    plt.show()

col_nos = [0, 1, 2, 3, 8, 9]
for i in col_nos:    
    fig, ax = plt.subplots()
    ax = plot_ma_on_ts(i, plug_sc, tp, ax)
    plt.show()

###############################################################################
# Plot all important time average together
###############################################################################
def plot_multiple_ma(col_nos, td, df):
    y_min, y_max = -2.5, 2.5
    ma_df = df.rolling(td).mean()
    for i in col_nos:
        colname = ma_df.columns[i]
        ax.plot(ma_df.iloc[:,i], lw = 0, marker = 'o', ms = 0.03, label = colname)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='x', rotation=90)    
    ax.legend(numpoints = 1, markerscale = 200)
    return ax

#col_nos = [1, 2, 3, 0]
col_nos = [0, 9]
col_nos = [1, 2]
col_nos = [2, 3]
col_nos = [1, 3]
td = '5d'
fig, ax = plt.subplots()
ax = plot_multiple_ma(col_nos, td, plug_sc)

col_nos = [1, 2, 3, 0]
#col_nos = [0, 9]
td = '5d'
fig, ax = plt.subplots()
ax = plot_multiple_ma(col_nos, td, abnormal_sc)


###############################################################################
# Plotting the normalized data
###############################################################################
sorted_df = abnormal_df.sort_values(by = 'Unnamed: 0')
#sorted_df = plug_df.sort_values(by = 'Unnamed: 0')
scaler = StandardScaler()
np_scaled = scaler.fit_transform(sorted_df.iloc[:,1:])

x = sorted_df.iloc[:,0]
cols = [1, 2,3,4,9, 10]
for i in cols:
    colname = sorted_df.columns[i]
    y = sorted_df.iloc[:,i].values
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    plt.plot(x,y_scaled, lw = 0, marker = 'o', ms = 0.03) 
    plt.title(colname)
    plt.ylim(-2.5,2.5)
    plt.xticks(rotation='vertical')
    plt.show()

###############################################################################
# determine the normal running period for fault detection algorithm
###############################################################################
abnormal_df.head()

abnormal_df.columns
abnormal_df.iloc[:,0].min()
abnormal_df.iloc[:,0].max()

abnormal_df[3]


###############################################################################
# dealing with smaller data frame
###############################################################################
df = abnormal_df.query('ilevel_0==2')

df.plot(x= 0, y = 1)
df_1 = df.where(df['FC8215LD.CPV'] > 80)

df.plot(x=0, y = 1)
df_1.plot(x=0, y = 1)


start_date = '2018-09-24 00:00:00'
end_date = '2018-11-07 00:00:00'

ind1 = abnormal_df.iloc[:,0] > start_date
ind2 = abnormal_df.iloc[:,0] < end_date
ind = ind1 & ind2

x = abnormal_df.loc[ind,:].iloc[:,0]
y = abnormal_df.loc[ind,:].iloc[:,1]

plt.plot(x, y, lw = 0, marker = 'o', ms = 0.03)
plt.xticks(rotation='vertical')
plt.title('Chosen')

###############################################################################
# data subsetting and visualization based on chosen data
###############################################################################
abnormal_df.columns

np.sum(ind)
col_ind = [1,2,3,4,10]

x = abnormal_df.loc[ind,:].iloc[:,0]
for i in col_ind:
    y = abnormal_df.loc[ind,:].iloc[:,i]
    plt.plot(x, y, lw = 0, marker = 'o', ms = 0.03)
    plt.xticks(rotation='vertical')
    plt.title(abnormal_df.columns[i])
    plt.show()


###############################################################################
# Data smoothening before modelling
###############################################################################
tp = 5*24*60
col_ind = [1,2,3,4,10]
subset_df = abnormal_df.loc[ind].iloc[:,col_ind]
ma_df = subset_df.rolling(tp).mean()
ma_df = ma_df.dropna(axis = 0, how = 'any')
for i in range(5):
    plt.plot(ma_df.iloc[:,i].values)
    plt.title(ma_df.columns[i])
    plt.show()
    
###############################################################################
# autoencoder modelling
###############################################################################
    
    
###############################################################################
# alarm setting input data
###############################################################################

abnormal_df.head()
i = 1
x = abnormal_df.iloc[:,0]
y = abnormal_df.iloc[:,i]


plt.plot(x, y, lw = 0, marker = 'o', ms = 0.03)

abnormal_df.loc[2]

###############################################################################
# dates
###############################################################################
start_date = '2018-04-12 00:00:00'
end_date = '2018-12-07 00:00:00'

loads = [70, 120]
margin = pd.Timedelta('5 days')
#start_date = '2014-04-12 00:00:00'
#end_date = '2015-12-07 00:00:00'
change_location_ind = get_load_change_index_locations_and_type(abnormal_df, loads)
valid_dates = get_all_start_end_time_within_specified_load_limit(change_location_ind, margin, abnormal_df)
load_indices = get_load_indices(valid_dates, abnormal_df)
date_indices = get_date_indices(start_date, end_date, abnormal_df)
indices = date_indices & load_indices

i = 1
x, y = get_df_subset(i, indices, abnormal_df)
plt.plot(x, y, lw = 0, marker = 'o', ms = 0.03)
plt.xticks(rotation='vertical')

plt.plot()



def get_load_change_index_locations_and_type(df, load):
    '''when the plant is running within the specified load
    the condition is true, otherwise it is false. Therefore,
    the difference value of +1 indicates the plant has moved into 
    the desired load zone. Conversley, the difference value of -1
    indicates that the plant has moved out of the desired load zone.
    '''
    print(load)
    ind1 = df.iloc[:,1] > load[0]
    ind2 = df.iloc[:,1] < load[1]
    ind = ind1 & ind2
    ind_int = ind*1
    change_point = ind_int.diff()
    change_ind = abs(change_point) == 1
    return change_point.loc[change_ind]

df = abnormal_df.query('ilevel_0==2')



plt.plot(df.iloc[:,0], df.iloc[:,1])

# flatten the data frame when we want to keep the index
df.reset_index()
# another option when we do not want the new column for the index
df = df.reset_index(drop=True)

loads = [90, 150]
change_indices = get_load_change_index_locations_and_type(df, loads)

i = 1
fig, ax = plt.subplots()
ax = plot_ts(i, df, ax, col = 'b')
ax = add_change_points(change_indices, df, ax)


####
loads = [90, 110]
margin = pd.Timedelta('5 days')
change_location_ind = get_load_change_index_locations_and_type(df, loads)
valid_dates = get_all_start_end_time_within_specified_load_limit(change_location_ind, margin, df)
load_indices = get_load_indices(valid_dates, df)

i = 1
fig, ax = plt.subplots()
ax = plot_subset_by_boolean_index(i, load_indices, df, ax, col = 'b', invert = False)
ax = plot_subset_by_boolean_index(i, load_indices, df, ax, col = 'r', invert = True)

###############################################################################
# flattening and sorting the data frame
###############################################################################

abnormal_df.head()
df = abnormal_df.rename(index=str, columns={'Unnamed: 0': 'datetime'})
df = df.reset_index(drop=True)
df = df.sort_values(by = ['datetime'])

###############################################################################
# plotting the time series
###############################################################################
def plot_ts(i, df, ax, col = 'b'):
    x = df.iloc[:,0]
    y = df.iloc[:,i]
    ax.plot(x, y, col, lw = 0, marker = 'o', ms = 0.03)
    return ax

def plot_subset_by_boolean_index(i, indices, df, ax, col = 'b', invert = False):
    x = df.iloc[:,0]
    y = df.iloc[:,i].copy(deep = True)
    if invert:
        inv_ind = indices
    else:
        inv_ind = np.invert(indices)
    y[inv_ind] = None
    ax.plot(x, y, col, lw = 0, marker = 'o', ms = 0.03)
    return ax


fig, ax = plt.subplots()
ax = plot_ts(1, df, ax)


###############################################################################
# getting the required data
###############################################################################
loads = [80, 150]
margin = pd.Timedelta('1 days')

start_date = '2018-04-01 00:00:00'
end_date = '2019-01-01 00:00:00'

change_location_ind = get_load_change_index_locations_and_type(df, loads)
valid_dates = get_all_start_end_time_within_specified_load_limit(change_location_ind, margin, df)
load_indices = get_load_indices(valid_dates, df)

date_indices = get_date_indices(start_date, end_date, df)
indices = date_indices & load_indices

i = 2
fig, ax = plt.subplots()
ax = plot_subset_by_boolean_index(i, load_indices, df, ax, col = 'b', invert = False)
ax = plot_subset_by_boolean_index(i, load_indices, df, ax, col = 'r', invert = True)
ax = plot_subset_by_boolean_index(i, indices, df, ax, col = 'g', invert = False)

fig, ax = plt.subplots()
ax = plot_subset_by_boolean_index(i, indices, df, ax, col = 'g', invert = False)
ax = plot_subset_by_boolean_index(i, indices, df, ax, col = 'r', invert = True)

###############################################################################
# time series smoothening
###############################################################################

df.head()
df = df.set_index('datetime')
df_ma = df.rolling('1d', min_periods = 24*60).mean()

i = 2
fig, ax = plt.subplots()
plot_ts(i, df_ma, ax, col = 'b')
plt.plot(df_ma.index, df_ma.iloc[:,0])

plt.plot(df.index, df.iloc[:,1])
plt.plot(df_ma.index, df_ma.iloc[:,1])

###############################################################################
# First subsetting based on load then 
###############################################################################

np.sum(indices)
df.where(indices, axis = 'columns')
df_sub = df.where(indices, axis = 1)

###############################################################################
# simple load test
###############################################################################
start_date = '2018-06-01 00:00:00'
end_date =   '2018-06-10 00:00:00'

test_df = df.loc[ : , ['FC8215LD.CPV', 'PI8221.PV', 'PI8228.PV', 'TI8221A.PV', 'FC8228D.PV']]

ind1 = test_df['FC8215LD.CPV'] > 80
ind2 = test_df['FC8215LD.CPV'] < 150
ind = ind1 & ind2


test_sub_df = test_df.where(ind, axis = 'columns')
test_sub_df.plot()
test_sub_ma = test_sub_df.rolling('1d', min_periods = 24*60).mean()


test_sub_df.plot()
test_sub_ma.plot()

without_na = test_sub_ma.dropna(axis = 'index', how = 'any')
without_na.head()

# try normalization without removing na values
std_scaler = StandardScaler()
np_scaled = std_scaler.fit_transform(test_sub_ma)



###############################################################################
# normalizing data
###############################################################################
std_scaler = StandardScaler()
np_scaled = std_scaler.fit_transform(without_na)
df_normalized = pd.DataFrame(np_scaled)
df_normalized.columns = without_na.columns
df_normalized

index_name = without_na.index.values
df_normalized['datetime'] = index_name
df_normalized.set_index('datetime', drop = True)
df_normalized.plot()


def normalize(x):
    mean = np.mean(x)
    sd = np.std(x)
    y = (x - mean)/sd
    return y

normalize_df = test_sub_ma.apply(normalize)
normalize_df.plot()

###############################################################################
# normalizing manually
###############################################################################



test_sub_ma.apply()

x = test_sub_ma.iloc[:,1] 


























