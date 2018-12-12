# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 08:56:46 2018

@author: Ravi Tiwari
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy.stats as stats
import seaborn as sns

###############################################################################
# Set the working directory
###############################################################################
os.chdir('C:\\Users\\40204945\\Documents\\sumitomo')
os.listdir('.')

###############################################################################
###############################################################################
# Read the data
###############################################################################
def read_plug_data():    
    col_no = range(5,18)
    plug_data = pd.read_excel('data\\IOT\\SMM1 T-1220 Plugging.xlsx',
                                 usecols = col_no, skiprows = 3)
    return plug_data

plug_data = read_plug_data()


###############################################################################
# Exploratory Analysis
###############################################################################
###############################################################################
# Finding the distribution
###############################################################################
def get_numeric_index(df, i):
    ind = []
    for j, n in enumerate(df.iloc[:,i]):
        if np.isreal(n):
            if not np.isnan(n):
                ind.append(j)
    return ind

def get_x(df, ind):
    x = df.iloc[ind,0]
    return x
    
def get_y(df, ind, i = 2):
    y = df.iloc[ind,i]
    return y
    
def plot_ts(i, df, desc):
    ind = get_numeric_index(df, i)
    col_name = df.columns[i]
    x = get_x(df, ind)
    y = get_y(df, ind, i)
    plt.plot(x,y) 
    plt.xticks(rotation = 'vertical')
    plt.title(col_name + ' : ' + desc[i-1])    
    fname = 'results\\figures\\SMM' + col_name.replace('.', '') + '_ts' + '.jpeg'
    plt.xlabel('Date (YYYY-MM)')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return

def plot_hist(i, bins, df, desc):
    ind = get_numeric_index(df, i)
    col_name = df.columns[i]
    y = get_y(df, ind, i)
    freq, edges = np.histogram(y.values, bins=bins)        
    plt.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge")
    plt.title(col_name + ' : ' + desc[i-1])
    fname = 'results\\figures\\SMM' + col_name.replace('.', '') + '_hist' + '.jpeg'
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return
    
###############################################################################
# Plotting abnormality data
###############################################################################
def plot_abnormality_data(df, desc):   
    for i in range(1,12):
        plot_ts(i, df, desc)
        plot_hist(i, 20, df, desc)
    return

def get_col_desc():
    col_desc = ['EXTRACTION CURRENT LOAD',	'T-8220 PRESS.',
  'E-8221A/B/C INLET SM', 'T-8220 BTM TEMP.',
  'E-8223 OUTLET LINE',	'E-8223 OUTLET LINE',
  'T-8220 BTM TEMP.', 'T-8220 BTM',	
  'E-8221A INLET SM', 'E-8221A/B/C SM',
   'E-8221A/B/C INLET OX',	'E-8221A/B INLET LINE']
    return col_desc
    
col_desc = get_col_desc()
plot_abnormality_data(plug_data, col_desc)


###############################################################################
# investigate load and bottom pressure
###############################################################################
###############################################################################
# Cross correlation
###############################################################################
def plot_moving_average_and_load(i , mv_period = 24, df = plug_data):
    valid_ind = get_numeric_index(df, i)
    time = get_x(df, valid_ind)
    col_1 = df.iloc[valid_ind, 1]
    col_2 = df.iloc[valid_ind, i]
    ma_2 = col_2.rolling(window = mv_period*60).mean()
    title = df.columns[i]
    y_label = 'E-8221 A/B/C Inlet SM (kPa)'
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_ylabel('Extraction Current Load', color=color)
    ax1.plot(time, col_1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation='vertical')
    ax2 = ax1.twinx()  
        
    color = 'tab:green'
    ax2.plot(time, ma_2, color=color)
    ax2.set_ylabel(y_label, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(title)
    fig.tight_layout()  
    plt.show()
    return
    


def plot_moving_average(i, mv_period, df):
    valid_ind = get_numeric_index(df, i)
    time = get_x(df, valid_ind)
    col_2 = df.iloc[valid_ind, i]
    ma_2 = col_2.rolling(window = mv_period*60).mean()
    title = df.columns[i]
    
    plt.plot(time, col_2)
    plt.plot(time, ma_2, color = 'tab:green', label = 'Moving Average')
    plt.xticks(rotation = 'vertical')
    plt.title(title)
    plt.legend()
    plt.show()
    return

plot_ts(i = 3, df = plug_data, desc = col_desc)
plot_moving_average(i = 3, mv_period = 1*24, df = plug_data)
plot_moving_average_and_load(i = 3 , mv_period = 1*24, df = plug_data)



###############################################################################
# Understand the load pattern
###############################################################################
plug_data.columns
plot_ts(i = 1, df = plug_data, desc = col_desc)
plot_hist(i = 1, bins = 20, df = plug_data, desc = col_desc)

ind = get_numeric_index(plug_data, 1)


plug_data.columns
plug_data.loc[ind,'FC1215LD.CPV'].pct_change().plot()
change = plug_data.loc[ind,'FC1215LD.CPV'].diff()
load = plug_data.loc[ind,'FC1215LD.CPV']


ch_ind = abs(change) < 5
ld_ind = load > 80
ind = ch_ind & ld_ind


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(load[ind], color=color)
ax2 = ax1.twinx()  
    
color = 'tab:green'
ax2.plot(change[ind], color=color)
fig.tight_layout()  
plt.show()


plt.plot(change[ch_ind])
plt.plot(load[ld_ind])
plt.plot(load[ch_ind])
plt.plot(load)
plt.plot(load[ind])


ch_ind & ld_ind

###############################################################################
# plot zoomed
###############################################################################
def plot_ts_zoomed(i, start_time, end_time, df, desc):    
    ind1 = df.iloc[:,0] > start_time
    ind2 = df.iloc[:,0] < end_time
    ind_time = ind1 & ind2
    
    df = df.loc[ind_time, :]
    
    ind = get_numeric_index(df, i[0])

    x = df.iloc[ind,0]
    y1 = get_y(df, ind, i[0])
    y1_label = plug_data.columns[i[0]] + ' : ' + desc[i[0] - 1]
    y2 = get_y(df, ind, i[1])
    y2_label = plug_data.columns[i[1]] + ' : ' + desc[i[1] - 1]
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_ylabel(y1_label, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation='vertical')
    ax2 = ax1.twinx()  
        
    color = 'tab:green'
    ax2.plot(x, y2, color=color)
    ax2.set_ylabel(y2_label, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout() 
    plt.xlim(start_time, end_time)    
    plt.show()
        
        
start_time = '2016-08-23 00:00:00'
end_time = '2016-08-23 03:59:00'
#end_time = '2018-02-23 23:59:00'
#i = [3, 4]  # inlet steam, bottom temperature
#i = [3, 2]  # inlet steam, column pressure
#i = [3, 8]  # inlet steam, bottom level
#i = [3, 11] # inlet steam, inlet ox
#i = [5, 6]  # reflux PC, reflux PI
i = [3, 5]    # inlet steam, reflux PC
#i = [3, 10]
plot_ts_zoomed(i, start_time, end_time, plug_data, col_desc)

###############################################################################
# 
###############################################################################
col_desc
plug_data.columns

for i, col in enumerate(col_desc):
    print(i+1, col)


###############################################################################
# Alarm limit setting using moving average
############################################################################### 
maval = plug_data.iloc[:,1].rolling(24*60).mean() 
plug_data.iloc[:,1].rolling(24*60)

valid_ind = get_numeric_index(plug_data, 2)
time = get_x(plug_data, valid_ind)
col_1 = plug_data.iloc[valid_ind, 1]
col_2 = plug_data.iloc[valid_ind, 2]
ma_2 = col_2.rolling(window = 24*60).mean()
ma_2
plt.plot(time, ma_2)
plt.xticks(rotation = 'vertical')
plt.plot(col_1, col_2)

###############################################################################
# Find out time when to when
###############################################################################


time
col_2.isnull().sum()
valid_ind = ma_2.notna()
time = time.loc[valid_ind]
col_2 = col_2.loc[valid_ind]

#plt.hist(col_2.values)
col_2.values.reshape(1,-1)[0]
x = np.random.normal(100, 100, size = 100)

sns.distplot(col_2.values)
sns.distplot(col_2.values.reshape(1,-1)[0])

x = np.array(col_2.values, dtype='float')
sns.distplot(x, bins = 300, color = 'red')
plt.xlim(150, 200)

###############################################################################
# data preprocessing
###############################################################################






















