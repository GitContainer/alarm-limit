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
from sys import getsizeof


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
# Step 1: Read the data
###############################################################################
#abnormal_df = get_cleaned_and_merged_df('abnormality_data')
#plug_df = get_cleaned_and_merged_df('plug_data')


###############################################################################
# pickle to loaded df
###############################################################################
#abnormal_df.to_pickle("./abnormal_df.pkl")
#plug_df.to_pickle("./plug_df.pkl")

###############################################################################
# Step 1: Read the pickled data
###############################################################################
abnormal_df = pd.read_pickle("./abnormal_df.pkl")
plug_df = pd.read_pickle("./plug_df.pkl")
abnormal_df.memory_usage()
getsizeof(abnormal_df)
getsizeof(plug_df)
###############################################################################    
###############################################################################
# data subsetting for alarm limit setting
###############################################################################
def get_load_change_index_locations_and_type(df, load):
    '''when the plant is running within the specified load
    the condition is true, otherwise it is false. Therefore,
    the difference value of +1 indicates the plant has moved into 
    the desired load zone. Conversley, the difference value of -1
    indicates that the plant has moved out of the desired load zone.
    '''
    ind1 = df.iloc[:,1] > load[0]
    ind2 = df.iloc[:,1] < load[1]
    ind = ind1 & ind2
    ind_int = ind*1
    change_point = ind_int.diff()
    change_ind = abs(change_point) == 1
    return change_point.loc[change_ind]

loads = [90, 100]
change_indices = get_load_change_index_locations_and_type(abnormal_df, loads)
###############################################################################
# Get all the start and end date with margin when the plant is running 
# within specified load
###############################################################################

def get_individual_start_and_end_time_within_specified_load_limit(i, change_point, margin, df):
    if i == 0:
        start_date = df.iloc[0,0]
        end_ind = change_point.index[0]
        end_date = df.loc[end_ind].iloc[0]
    else:
        start_ind = change_point.index[i]
        start_date = df.loc[start_ind].iloc[0]
        try:
            end_ind = change_point.index[i+1]
        except:
            end_date = df.iloc[-1,0]
        else:
            end_date = df.loc[end_ind].iloc[0]
    return start_date + margin, end_date - margin
    
    
def get_all_start_end_time_within_specified_load_limit(change_points, margin, df):
    dates = []
    for i, item in enumerate(change_points):
        if i == 0:
            if item == -1:              
                date = get_individual_start_and_end_time_within_specified_load_limit(i, change_points, margin, df)
                dates.append(date)
        if item == 1:           
            date = get_individual_start_and_end_time_within_specified_load_limit(i, change_points, margin, df)
            dates.append(date)
    return dates
    
###############################################################################
# Get boolean indices when the plant is running within specified load and within
# specified period
###############################################################################

def get_load_indices(valid_dates, df):
    n, _ = df.shape
    load_indices = np.zeros(n, dtype = bool)
    for date1, date2 in valid_dates:
        ind1 = df.iloc[:,0] > date1
        ind2 = df.iloc[:,0] < date2
        ind = ind1 & ind2
        load_indices = load_indices | ind
    return load_indices
        
def get_date_indices(start_date, end_date, df):
    ind1 = df.iloc[:,0] > start_date
    ind2 = df.iloc[:,0] < end_date
    ind = ind1 & ind2 
    return ind

def get_indices(load_indices, date_indices):
    indices = date_indices & load_indices
    return indices
    

###############################################################################
# Subset data frame based on indices
###############################################################################
def get_df_subset(i, ind, df):
    x = df.loc[ind,:].iloc[:,0].values
    y = df.loc[ind,:].iloc[:,i].values.copy()
    y = np.array(y, dtype='float')
    return x, y
    

###############################################################################
# Get statistics
###############################################################################
def get_mean_sd(y):     
    mean = np.mean(y)
    sd = np.std(y)
    return mean, sd    

def moving_average(y, window=24*60) :
    ret = np.cumsum(y, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


###############################################################################
#  get the result
###############################################################################
loads = [90, 100]
margin = pd.Timedelta('5 days')
start_date = '2014-04-12 00:00:00'
end_date = '2015-12-07 00:00:00'
change_indices = get_load_change_index_locations_and_type(abnormal_df, loads)
valid_dates = get_all_start_end_time_within_specified_load_limit(change_indices, margin, abnormal_df)
load_indices = get_load_indices(valid_dates, abnormal_df)
date_indices = get_date_indices(start_date, end_date, abnormal_df)
indices = get_indices(load_indices, date_indices)

i = 4
x, y = get_df_subset(i, indices, abnormal_df)
x1, y1 = get_df_subset(i, date_indices, abnormal_df)
x2, y2 = get_df_subset(i, load_indices, abnormal_df)
mean, sd = get_mean_sd(y)
y_mv = moving_average(y, window=24*60)
plt.plot(x,y)
plt.plot(x[24*60 - 1:], y_mv)












