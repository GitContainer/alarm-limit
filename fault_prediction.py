# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:11:16 2019

@author: Ravi Tiwari
"""

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
scale_dic['plug']= {}
scale_dic['plug']['FC1215LD.CPV'] = [60, 115]
scale_dic['plug']['TI1221A.PV'] = [115, 129]
scale_dic['plug']['PI1221.PV'] = [160, 260]
scale_dic['plug']['FC1228c.PV'] = [3.5, 5.5]
scale_dic['plug']['FI1228A.PV'] = [1.2, 3.2]
scale_dic['plug']['PI1228.PV'] = [150, 280]


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
# Plotting the abnormal trends
###############################################################################










###############################################################################
# Distillation column bottom level, Relux pressure and the reboiler chemical 
# dosing are not indicative
###############################################################################
# 6. Distillation column bottom level
i = 8
y = sorted_df.iloc[:,i]
plt.plot(x,y, lw = 0, marker = 'o', ms = 0.03)
#plt.ylim(66, 73)
plt.ylim(55, 85)
plt.title(colnames[i])
plt.xticks(rotation='vertical')

# 7. Reflux pressure Controller
i = 5
y = sorted_df.iloc[:,i]
plt.plot(x,y, lw = 0, marker = 'o', ms = 0.03)
#plt.ylim(104, 106)
plt.ylim(100, 110)
plt.title(colnames[i])
plt.xticks(rotation='vertical')

# 8. Reflux pressure Indicator
i = 6
y = sorted_df.iloc[:,i]
plt.plot(x,y, lw = 0, marker = 'o', ms = 0.03)
#plt.ylim(106, 108)
plt.ylim(100, 110)
plt.title(colnames[i])
plt.xticks(rotation='vertical')

# 9. reboiler chemical dosing
i = 11
y = sorted_df.iloc[:,i]
plt.plot(x,y, lw = 0, marker = 'o', ms = 0.03)
#plt.ylim(38.50, 39.50)
plt.ylim(21, 25)
plt.title(colnames[i])
plt.xticks(rotation='vertical')





































