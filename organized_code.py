# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 09:14:28 2018

@author: Ravi Tiwari
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy.stats as stats
import seaborn as sns
from io import StringIO


###############################################################################
# Set the working directory
###############################################################################
os.chdir('C:\\Users\\40204945\\Documents\\sumitomo')
os.listdir('.')

###############################################################################
###############################################################################
# Read the data
###############################################################################
def read_xl_data(i, name):    
    col_no = range(5,18)
    if name == 'plug_data':
        fname = 'data\\IOT\\shared on 13.12.18\\SMM1 T-1220 Plugging.xlsx'
    if name == 'abnormality_data':
        fname = 'data\\IOT\\shared on 13.12.18\\Abnormality Detection May17 _ Jan18.xlsx'
        
    df = pd.read_excel(fname, usecols = col_no, skiprows = 3, sheet_name=i)    
    return df

def remove_non_numeric_values(df):
    n, _ = df.shape
    ind = []
    for i in range(n):
        row_vals = df.iloc[i,:].values
        ind_list = [np.isreal(x) for x in row_vals]
        ind.append(all(ind_list))
    cleaned_df = df.loc[ind,:]
    return cleaned_df


def get_cleaned_and_merged_df(name):
    df_0 = read_xl_data(0, name)
    df_1 = read_xl_data(1, name)
    df_2 = read_xl_data(2, name)
    
    cleaned_df_0 = remove_non_numeric_values(df_0)
    cleaned_df_1 = remove_non_numeric_values(df_1)
    cleaned_df_2 = remove_non_numeric_values(df_2)
    
    frames = [cleaned_df_0, cleaned_df_1, cleaned_df_2]
    cleaned_df = pd.concat(frames, keys=[0,1,2])
    return cleaned_df

###############################################################################    
###############################################################################
# data subsetting for alarm limit setting
###############################################################################
def get_change_point_location(df, load):
    ind1 = df.iloc[:,1] > load[0]
    ind2 = df.iloc[:,1] < load[1]
    ind = ind1 & ind2
    ind_int = ind*1
    change_point = ind_int.diff()
    change_ind = abs(change_point) == 1
    return change_point.loc[change_ind]


def get_index(start_date, end_date, margin, df):
    ind1 = df.iloc[:,0]  > start_date + margin
    ind2 = df.iloc[:,0]  < end_date - margin
    ind = ind1 & ind2
    return ind


def get_start_end_date_running_plant(change_point, df):
    start_date = df.iloc[0,0]
    try:
       end_ind = change_point.index[0] 
    except:
        end_date = df.iloc[-1,0]
    else:
        end_date = df.loc[end_ind].iloc[0]
    return start_date, end_date


def get_start_end_date_plant_restart(i, change_point, df):
    start_ind = change_point.index[i]
    start_date = df.loc[start_ind].iloc[0]
    try:
        end_ind = change_point.index[i+1]
    except:
        end_date = df.iloc[-1,0]
    else:
        end_date = df.loc[end_ind].iloc[0]
    return start_date, end_date

def get_indices_for_normal_period(change_point, margin, df):
    n, _ = df.shape
    ind = np.zeros(n, dtype = bool)
    for i, item in enumerate(change_point):
        if i == 0:
            if item == -1:              
                start_date, end_date = get_start_end_date_running_plant(change_point, df) 
                c_ind = get_index(start_date, end_date, margin, df)
                ind = ind | c_ind
        if item == 1:           
            start_date, end_date = get_start_end_date_plant_restart(i, change_point, df)
            c_ind = get_index(start_date, end_date, margin, df)
            ind = ind | c_ind
    return ind
        
def subset_data_for_alarm_limit_setting(i, ind, df, dates = None):
    if dates is not None:
        ind1 = df.iloc[:,0] > dates[0]
        ind2 = df.iloc[:,0] < dates[1]
        date_ind = ind1 & ind2
    
        x = df.loc[date_ind,:].iloc[:,0].values
        y = df.loc[date_ind,:].iloc[:,i].values.copy()
        f_ind = date_ind & ind
        y_clean = df.loc[f_ind,:].iloc[:,i].values.copy()
    else:
        x = df.iloc[:,0].values
        y = df.iloc[:,i].values.copy()
        y_clean = df.loc[ind,:].iloc[:,i].values.copy()
    
    y = np.array(y, dtype='float')
    y_clean = np.array(y_clean, dtype='float')
    return x, y, y_clean 


def get_input_data_for_alarm_setting(i, load, margin, df, dates):
    indices = []
    input_data = []
    try:
        levels = df.index.levels[0]
    except:
        pass
    else:
        for level in levels:
            sdf = df.loc[level]
            change_point = get_change_point_location(sdf, load)
            ind = get_indices_for_normal_period(change_point, margin, sdf)
            raw_data = subset_data_for_alarm_limit_setting(i, ind, sdf, dates) #x, y, y_clean
            indices.append(ind)
            input_data.append(raw_data)
    return input_data, indices

###############################################################################
# Get statistics
###############################################################################
def get_mean_sd(input_data):
    
    y_clean = []
    for item in input_data:
        y_clean.extend(item[2])
        
    y_clean = np.array(y_clean)
    
    mean = np.mean(y_clean)
    sd = np.std(y_clean)

    return mean, sd    

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


###############################################################################
# Step 1: Read the data
###############################################################################
abnormal_df = get_cleaned_and_merged_df('abnormality_data')
plug_df = get_cleaned_and_merged_df('plug_data')

###############################################################################
# Step 2: get the subset data for alarm limit setting
###############################################################################
i = 4
print(cleaned_abnormal_df.columns[i])
load = [90, 100]
start_date = '2014-04-12 00:00:00'
end_date = '2015-12-07 00:00:00'
dates = [start_date, end_date]
margin = pd.Timedelta('5 days')
abnormal_input_data, indices = get_input_data_for_alarm_setting(i, load, margin, cleaned_abnormal_df, dates)
mean, sd = get_mean_sd(abnormal_input_data)







###############################################################################
# Step 3: Check the input data by plotting
###############################################################################
df = cleaned_abnormal_df.loc[0]
ind1 = df.iloc[:,0] > start_date
ind2 = df.iloc[:,0] < end_date
date_ind = ind1 & ind2
f_ind = date_ind & ind

x = df.loc[date_ind,:].iloc[:,0].values
y = df.loc[date_ind,:].iloc[:,i].values.copy()
y_clean = df.loc[f_ind,:].iloc[:,i].values.copy()

# debuggin
i = 4
print(cleaned_abnormal_df.columns[i])
load = [90, 100]
start_date = '2014-04-12 00:00:00'
end_date = '2015-06-07 00:00:00'
dates = [start_date, end_date]
margin = pd.Timedelta('5 days')
abnormal_input_data, indices = get_input_data_for_alarm_setting(i, load, margin, cleaned_abnormal_df, dates)
ind = indices[0]
change_point = get_change_point_location(cleaned_abnormal_df.loc[0], load)


i = 4
start_date = '2014-04-12 00:00:00'
end_date = '2015-06-07 00:00:00'
df = cleaned_abnormal_df
indices = []
input_data = []
for j in range(3):
    sdf = df.loc[j]
    change_point = get_change_point_location(sdf, load)
    ind = get_indices_for_normal_period(change_point, margin, sdf)
    raw_data = subset_data_for_alarm_limit_setting(i, ind, sdf, dates) #x, y, y_clean
    indices.append(ind)
    input_data.append(raw_data)



len(input_data)
for item in input_data:
    print(len(item[0]), len(item[1]), len(item[2]))




###############################################################################
# Step 3: Plot the results
###############################################################################
for i, item in enumerate(abnormal_input_data):
    x, y = item[0], item[1]
    if i == 0:
        ax = plot_alarm_limit_on_ts(x, y, mean, sd)
    else:
        ax = plot_alarm_limit_on_ts(x, y, mean, sd, ax)
    











