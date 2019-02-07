# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 09:14:28 2018

@author: Ravi Tiwari
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

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
    print(load)
    ind1 = df.iloc[:,1] > load[0]
    ind2 = df.iloc[:,1] < load[1]
    ind = ind1 & ind2
    ind_int = ind*1
    change_point = ind_int.diff()
    change_ind = abs(change_point) == 1
    return change_point.loc[change_ind]


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
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    ind1 = df.iloc[:,0] > start_date
    ind2 = df.iloc[:,0] < end_date
    ind = ind1 & ind2 
    return ind

###############################################################################
# Careful same name function exists in the organized_plotting code
###############################################################################
#def get_indices(load_indices, date_indices):
#    indices = date_indices & load_indices
#    return indices
def get_indices(dates, loads, margin, df):
    start_date, end_date = dates
    change_location_ind = get_load_change_index_locations_and_type(df, loads)
    valid_dates = get_all_start_end_time_within_specified_load_limit(change_location_ind, margin, df)
    load_indices = get_load_indices(valid_dates, df)
    date_indices = get_date_indices(start_date, end_date, df)
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

def get_alarm_setting(i, indices, df):
    
    tag_name = df.columns[i]
    
    y = df.loc[indices,:].iloc[:,i].values.copy()
    y = np.array(y, dtype='float')
    
    mean = np.mean(y)
    sd = np.std(y)
    
    hh = mean + 3*sd
    ll = mean - 3*sd
    
    hi = mean + 2*sd
    lo = mean - 2*sd
    
    return tag_name, mean, sd, hh, hi, lo, ll

def create_empty_df():
    col_names = ['tag', 'mean', 'std', 'high high', 'high', 'low', 'low low']
    df_alarm_setting = pd.DataFrame(columns = col_names)
    return df_alarm_setting

def create_dictionary(tag, mean, sd, hh, hi, lo, ll):
    values = {'tag': tag,
             'low':lo, 
             'high':hi,
             'mean':mean,
             'std':sd,
             'low low': ll,
             'high high': hh}
    return values


def add_alarm_values(df_alarm_setting, values):
    df_alarm_setting = df_alarm_setting.append(values, ignore_index = True)
    return df_alarm_setting    

def save_alarm_limits_in_df(df, indices):    
    df_alarm_setting = create_empty_df()
    _, n = df.shape
    
    for i in range(2, n):
        tag_name, mean, sd, hh, hi, lo, ll = get_alarm_setting(i, indices, df)
        values = create_dictionary(tag_name, mean, sd, hh, hi, lo, ll)
        df_alarm_setting = add_alarm_values(df_alarm_setting, values)
        
    return df_alarm_setting
        


###############################################################################
#  get the result
###############################################################################
# alarm setting for abnormal data

start_date = '2014-04-12'
end_date = '2015-12-07'
load_range = [90, 100]
date_range = [start_date, end_date]
margin = pd.Timedelta('5 days')

indices = get_indices(date_range, load_range, margin, abnormal_df)

abnormal_alarm_setting = save_alarm_limits_in_df(abnormal_df, indices)
abnormal_alarm_setting.round(2)
abnormal_alarm_setting.to_csv('abnormal_alarm_setting.csv', float_format='%.2f')

# alarm setting for plug data
start_date = '2016-07-01'
end_date = '2017-12-01'
load_range = [90, 110]
date_range = [start_date, end_date]
margin = pd.Timedelta('5 days')

indices = get_indices(date_range, load_range, margin, plug_df)

plug_alarm_setting = save_alarm_limits_in_df(plug_df, indices)
plug_alarm_setting.round(2)
plug_alarm_setting.to_csv('plug_alarm_setting.csv', float_format='%.2f')


###############################################################################
# individual result
###############################################################################
start_date = '2014-04-12'
end_date = '2015-12-07'
load_range = [90, 100]
date_range = [start_date, end_date]
margin = pd.Timedelta('5 days')

indices = get_indices(date_range, load_range, margin, abnormal_df)
tag_name, mean, sd, hh, hi, lo, ll =  get_alarm_setting(4, indices, abnormal_df) 
    





















