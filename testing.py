# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 11:26:21 2018

@author: Ravi Tiwari
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


###############################################################################
# data subsetting functions
###############################################################################
def get_subset_df(j, dates, df):
    ind1 = df.loc[j].iloc[:, 0] > dates[0]
    ind2 = df.loc[j].iloc[:, 0] < dates[1]
    ind = ind1 & ind2
    
    subset_df = df.loc[j].loc[ind].iloc[:,0:4]
    return subset_df


#def get_change_point_location(df, load):
#    ind = df.iloc[:,1] > load
#    ind_int = ind*1
#    change_point = ind_int.diff()
#    change_ind = abs(change_point) == 1
#    return change_point.loc[change_ind]


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
#        print(i, item)
        if i == 0:
            if item == -1:
#                print(i)                
                start_date, end_date = get_start_end_date_running_plant(change_point, df) 
#                print(start_date, end_date)
                c_ind = get_index(start_date, end_date, margin, df)
                ind = ind | c_ind
        if item == 1:
#            print(i)            
            start_date, end_date = get_start_end_date_plant_restart(i, change_point, df)
#            print(start_date, end_date)
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
    y_clean = np.array(y, dtype='float')
    return x, y, y_clean 

def get_y_stats(y):
    mean = np.mean(y)
    sd = np.std(y)
    return mean, sd

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


###############################################################################
# data subsetting
###############################################################################
start_date = '2014-06-21 00:00:00'
start_date = '2014-07-21 00:00:00'
end_date = '2015-07-30 00:00:00'
margin = pd.Timedelta('0 days')
dates = [start_date, end_date]
load = [90, 97]
sdf = get_subset_df(0, dates, cleaned_abnormal_df)

sdf = cleaned_abnormal_df.loc[0]
change_point = get_change_point_location(sdf, load)
ind = get_indices_for_normal_period(change_point, margin, sdf)
x, y, y_clean = subset_data_for_alarm_limit_setting(2, dates, ind, sdf)
mean, sd = get_y_stats(y_clean)




###############################################################################
# Plotting functions
###############################################################################
def plot_ts_subset(i, df):
    f, ax = plt.subplots()
    x = df.iloc[:,0]
    y = df.iloc[:,i]
    ax.plot(x, y, color = 'b') 
    plt.xticks(rotation = 'vertical') 
    return ax


def add_change_points_subset(change_point, df, ax):    
    ind = change_point.index
    x_change = df.loc[ind].iloc[:,0]
    for t, ct in zip(x_change, change_point):
        if ct == 1:
            col = 'red'
        if ct == -1:
            col = 'green'
        ax.axvline(x = t, color = col, linewidth = 1)    
    return ax

def set_x_limit(start_date, end_date, ax):
    ax.set_xlim(start_date, end_date)
    return ax   

def set_y_limit(low, high, ax):
    ax.set_ylim(low, high)
    return ax

def add_y_line(y, ax):
    ax.axhline(y = y, color = 'k', linewidth = 1, linestyle = '--')
    return ax



def plot_values_in_specified_load_and_margin(i, ind, df, ax = None):
    ylabel = df.columns[i]
    if ax is None:
        f, ax0 = plt.subplots()
    else:
        ax0 = ax.twinx()
    x = df.iloc[:,0]
    y = df.iloc[:,i].copy(deep = True)
    inv_ind = np.invert(ind)
    y[inv_ind] = None
    ax0.plot(x, y, color = 'orange')
    plt.xticks(rotation = 'vertical')
    ax0.tick_params(axis='y', labelcolor='orange')
    ax0.set_ylabel(ylabel, color='orange')
    return ax0

def plot_histogram_with_alarm_limits(y, mean, sd, ax = None):
    if ax is None:
        f, ax = plt.subplots()
    hh = round(mean + 2*sd, 2)
    ll = round(mean - 2*sd, 2)
    sns.distplot(y, bins = 30, color = 'green', vertical = True, ax = ax)
    ax.axhline(y = mean, color = 'k', linestyle = '--')
    ax.axhline(y = hh, color = 'red', linestyle = '--')
    ax.axhline(y = ll, color = 'red', linestyle = '--')
    ax.text(0.1, 0.9, 'HH: ' + str(hh), transform=ax.transAxes)
    ax.text(0.1, 0.8, 'LL: ' + str(ll), transform=ax.transAxes)
    return ax

def plot_alarm_limit_on_ts(x, y, mean, sd, ax = None):
    if ax is None:
        f, ax = plt.subplots()
        
    lower = mean - 2*sd
    upper = mean + 2*sd
    
    youtside = np.ma.masked_inside(y, lower, upper)
    yinside = np.ma.masked_outside(y, lower, upper)
    
    ax.plot(x, youtside, 'red', label = 'Abnormal')
    ax.plot(x, yinside, 'green', label = 'Normal')
         
    ax.axhline(y=lower, color = 'red', linestyle='--')
    ax.axhline(y=mean, color = 'k', linestyle='--')
    ax.axhline(y=upper, color = 'red', linestyle='--')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim(mean - 5*sd, mean + 5*sd)
#    ax.legend()
    return ax


def plot_moving_average(i, df, window_size = 1*24*60, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    
    x = df.iloc[:,0]
    y = df.iloc[:,i].copy(deep=True)
    y = np.array(y, dtype='float')
    y_mv = moving_average(y, n=window_size)
    ax.plot(x[window_size-1:], y_mv, color = 'red')    
    ax.tick_params(axis='x', rotation=90)
    return ax




###############################################################################
# plotting
###############################################################################
i = 3
sdf = cleaned_abnormal_df.loc[2]
#start_date = '2014-04-16 00:00:00'
#end_date = '2015-10-29 00:00:00'
#start_date = '2017-05-20 00:00:00'
#end_date = '2018-01-29 00:00:00'
start_date = '2018-04-12 00:00:00'
end_date = '2018-12-07 00:00:00'
dates = [start_date, end_date]
margin = pd.Timedelta('5 days')
load = [90, 100]

change_point = get_change_point_location(sdf, load)
ind = get_indices_for_normal_period(change_point, margin, sdf)

ax = plot_ts_subset(1, sdf)
ax = set_y_limit(80, 110, ax)
ax = add_y_line(load[0], ax)
ax = add_y_line(load[1], ax)
#ax = add_change_points_subset(change_point, sdf, ax)
ax0 = plot_values_in_specified_load_and_margin(i, ind, sdf, ax)
#ax0 = set_y_limit(150, 250, ax0)



change_point = get_change_point_location(sdf, load)
ind = get_indices_for_normal_period(change_point, margin, sdf)
x, y, y_clean = subset_data_for_alarm_limit_setting(i, dates, ind, sdf)
mean, sd = get_y_stats(y_clean)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
plot_alarm_limit_on_ts(x, y, mean, sd, ax1)
plot_histogram_with_alarm_limits(y_clean, mean, sd, ax2)

ax = plot_ts_subset(3, sdf)
ax = plot_moving_average(3, sdf, window_size = 5*24*60, ax = ax)
ax = set_y_limit(175, 230, ax)

###############################################################################
# Now implement for the whole system
###############################################################################
def get_change_points_and_merged_indices(margin, load, df):
    change_points = []
    merged_indices = []
    try:
        levels = df.index.levels[0]
    except:
        pass
    else:
        for level in levels:
            sdf = df.loc[level]            
            change_point = get_change_point_location(sdf, load)
            ind = get_indices_for_normal_period(change_point, margin, sdf)
            change_points.append(change_point)
            merged_indices.append(ind)
    return change_points, merged_indices


def get_input_data_for_alarm_setting(i, merged_indices, df, dates = None):
    input_data = []
    try:
        levels = df.index.levels[0]
    except:
        pass
    else:
        for level in levels:
            sdf = df.loc[level]
            ind = merged_indices[level]
            raw_data = subset_data_for_alarm_limit_setting(i, ind, sdf) #x, y, y_clean
            input_data.append(raw_data)
    return input_data


def get_mean_sd(input_data):
    y_clean = []
    for item in input_data:
        y_clean.extend(item[2])
    y_clean = np.array(y_clean)
    
    mean, sd = get_y_stats(y_clean)
    return mean, sd
    

margin = pd.Timedelta('2 days')
load = [90, 100]
change_points, merged_indices =  get_change_points_and_merged_indices(margin, load, cleaned_abnormal_df)
input_data = get_input_data_for_alarm_setting(2, merged_indices, cleaned_abnormal_df)
mean, sd = get_mean_sd(input_data)


for i, item in enumerate(input_data):
    x, y = item[0], item[1]
    if i == 0:
        ax = plot_alarm_limit_on_ts(x, y, mean, sd)
    else:
        ax = plot_alarm_limit_on_ts(x, y, mean, sd, ax)
    
        










