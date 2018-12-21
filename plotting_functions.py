# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:04:22 2018

@author: Ravi Tiwari
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

###############################################################################
# key variables
# cleaned_abnormal_df
# abnormal_change_points
# abnormal_merged_change_point
# margin = pd.Timedelta('10 days')
# abnormal_merged_indices
###############################################################################


def plot_ts(i, df, ax = None):    
    ylabel = df.columns[i]
    
    if ax is None:
        f, ax = plt.subplots()
            
    for j in range(3):
        x = df.loc[j].iloc[:,0]
        y = df.loc[j].iloc[:,i]
        ax.plot(x, y, color = 'b') 
        
    plt.xticks(rotation = 'vertical') 
    plt.ylabel(ylabel)
    return ax

def set_x_limit(start_date, end_date, ax):
    ax.set_xlim(start_date, end_date)
    return ax   

def set_y_limit(low, high, ax):
    ax.set_ylim(low, high)
    return ax

def add_y_line(y):
    ax.axhline(y = y, color = 'k', linewidth = 1, linestyle = '--')
    return ax

def add_change_points(change_point, df, ax):    
    for j in range(3):
        ind = change_point[j].index
        x_change = df.loc[j].loc[ind].iloc[:,0]
        for t, ct in zip(x_change, change_point[j]):
            if ct == 1:
                col = 'red'
            if ct == -1:
                col = 'green'
            ax.axvline(x = t, color = col, linewidth = 1)    
    return ax
    


def plot_values_in_specified_load_and_margin(i, merged_indices, df, ax = None):
    ylabel = df.columns[i]
    if ax is None:
        f, ax0 = plt.subplots()
    else:
        ax0 = ax.twinx()
    
    for j in range(3):
        ind = merged_indices[j]
        x = df.loc[j].iloc[:,0] 
        y = df.loc[j].iloc[:,i].copy(deep = True)
        inv_ind = np.invert(ind)
        y[inv_ind] = None
        ax0.plot(x, y, color = 'green')
    plt.xticks(rotation = 'vertical')
    ax0.tick_params(axis='y', labelcolor='green')
    ax0.set_ylabel(ylabel, color='green')
    return ax0





###############################################################################
# Plotting alarm limit
###############################################################################

def subset_data_for_alarm_limit_setting(i, dates, merged_indices, df):
    
    ind1 = df.iloc[:,0] > dates[0]
    ind2 = df.iloc[:,0] < dates[1]
    date_ind = ind1 & ind2
    
    x = df.loc[date_ind,:].iloc[:,0].values
    y = df.loc[date_ind,:].iloc[:,i].values.copy()
    y = np.array(y, dtype='float')
        
    ind = date_ind & merged_indices
    y_clean = df.loc[ind,:].iloc[:,i].values.copy()
    y_clean = np.array(y, dtype='float')
    return x, y, y_clean    

def get_y_stats(y):
    mean = np.mean(y)
    sd = np.std(y)
    return mean, sd

def plot_histogram_with_alarm_limits(y, mean, sd, ax = None):
    if ax is None:
        f, ax = plt.subplots()
    hh = round(mean + 3*sd, 2)
    ll = round(mean - 3*sd, 2)
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
        
    lower = mean - 3*sd
    upper = mean + 3*sd
    
    youtside = np.ma.masked_inside(y, lower, upper)
    yinside = np.ma.masked_outside(y, lower, upper)
    
    ax.plot(x, youtside, 'red', label = 'Abnormal')
    ax.plot(x, yinside, 'green', label = 'Normal')
         
    ax.axhline(y=lower, color = 'red', linestyle='--')
    ax.axhline(y=mean, color = 'k', linestyle='--')
    ax.axhline(y=upper, color = 'red', linestyle='--')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim(mean - 10*sd, mean + 10*sd)
    ax.legend()
    return ax




i = 2
start_date = '2014-04-16 00:00:00'
end_date = '2018-12-07 00:00:00'
margin = pd.Timedelta('10 days')

ax = plot_ts(1, cleaned_abnormal_df)
ax = add_y_line(90)
#ax = set_y_limit(86, 93, ax)

#ax = add_change_points(abnormal_change_points, cleaned_abnormal_df, ax)
#ax = add_merged_change_points_with_margin(abnormal_merged_change_point, df, margin, ax)
ax = set_x_limit(start_date, end_date, ax)
ax = set_y_limit(70, 110, ax)
ax = plot_values_in_specified_load_and_margin(i, abnormal_merged_indices, cleaned_abnormal_df, ax)
#ax = set_x_limit(start_date, end_date, ax)
#ax = set_y_limit(180, 280, ax)

dates = [start_date, end_date]
x, y, y_clean = subset_data_for_alarm_limit_setting(i, dates, abnormal_merged_indices, cleaned_abnormal_df)
mean, sd = get_y_stats(y_clean)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
plot_alarm_limit_on_ts(x, y, mean, sd, ax1)
plot_histogram_with_alarm_limits(y_clean, mean, sd, ax2)


###############################################################################
# moving average
###############################################################################
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_moving_average(i, df, window_size = 1*24*60, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    
    for j in range(3):
        x = df.loc[j].iloc[:,0]
        y = df.loc[j].iloc[:,i].copy(deep=True)
        y = np.array(y, dtype='float')
        y_mv = moving_average(y, n=window_size)
        ax.plot(x[window_size-1:], y_mv, color = 'red')
    
    ax.tick_params(axis='x', rotation=90)
    return ax

ax = plot_ts(2, cleaned_abnormal_df)
ax = plot_moving_average(2, cleaned_abnormal_df, window_size = 5*24*60, ax = ax)
ax = set_x_limit(start_date, end_date, ax)
ax = set_y_limit(200, 260, ax)


###############################################################################
# Needs some work
###############################################################################
def add_merged_change_points_with_margin(i, change_point, df, margin, ax):
    ylabel = df.columns[i]
    for j in range(3):
        ind = change_point[j].index
        x_change = df.loc[j].loc[ind].iloc[:,0]
        for t, ct in zip(x_change, change_point[j]):
            if ct == 1:
                col = 'red'
                ax.axvline(x = t + margin, color = col, linewidth = 1)
            if ct == -1:
                col = 'green'
                ax.axvline(x = t - margin, color = col, linewidth = 1)            
    plt.ylabel(ylabel)
    return ax
###############################################################################





