# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:34:22 2018

@author: Ravi Tiwari
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_ts(i, df, ax, col = 'b'):
    try:
        levels = df.index.levels[0]
        for level in levels:
            x = df.loc[level].iloc[:,0]
            y = df.loc[level].iloc[:,i]
            ax.plot(x, y, col)
    except:
        x = df.iloc[:,0]
        y = df.iloc[:,i]
        ax.plot(x, y, col)
    return ax

def rotate_x_label(ax, angle):
    ax.tick_params(axis='x', rotation=angle)
    return ax


def plot_subset_by_boolean_index(i, indices, df, ax, col = 'b', invert = False):
    try:
        levels = df.index.levels[0]
        for level in levels:
            ind = indices[level]
            x = df.loc[level].iloc[:,0]
            y = df.loc[level].iloc[:,i].copy(deep = True)
            if invert:
                inv_ind = ind
            else:
                inv_ind = np.invert(ind)            
            y[inv_ind] = None
            ax.plot(x, y, col)
    except:
        x = df.iloc[:,0]
        y = df.iloc[:,i].copy(deep = True)
        inv_ind = np.invert(indices)
        y[inv_ind] = None
        ax.plot(x, y, col)
    return ax
    
def create_twin_axis(ax):
    ax0 = ax.twinx()
    return ax0   
    
def set_x_limit(start_date, end_date, ax):
    ax.set_xlim(start_date, end_date)
    return ax       

def set_y_limit(low, high, ax):
    ax.set_ylim(low, high)
    return ax

def add_y_label(label, ax, col = 'k'):
    ax.set_ylabel(label, color = col)
    ax.tick_params('y', colors=col)
    return ax


def add_h_line(y, ax):
    for value in y:
        ax.axhline(y = value, color = 'k', linewidth = 1, linestyle = '--')
    return ax

def add_v_line(x, ax):
    for value in x:
        ax.axvline(x = value, color = 'k', linewidth = 1, linestyle = '--')
    return ax


def add_x_label(label, ax):
    ax.set_xlabel(label)
    return ax


def add_plot_title(ax, title):
    ax.set_title(title)
    return ax


def add_change_points(change_point, df, ax):    
    ind = change_point.index
    x_change = df.loc[ind].iloc[:,0]
    for t, ct in zip(x_change, change_point):
        if ct == 1:
            col = 'red'
        if ct == -1:
            col = 'green'
        ax.axvline(x = t, color = col, linewidth = 1, linestyle = '--')    
    return ax
    

###############################################################################
# items to be used in plotting
###############################################################################
# valid_dates, load_indices, date_indices, indices, start_date, end_date, abnormal_df
# start_date = 
###############################################################################


###############################################################################
# Histogram plot
###############################################################################   
def plot_histogram_with_alarm_limits(y, mean, sd, ax, vertical = True):
    hh = round(mean + 3*sd, 2)
    ll = round(mean - 3*sd, 2)
    if vertical:
        sns.distplot(y, bins = 30, color = 'green', vertical = True, ax = ax)
        ax.text(0.8, 0.9, 'HH: ' + str(hh), transform=ax.transAxes)
        ax.text(0.8, 0.8, 'LL: ' + str(ll), transform=ax.transAxes)
        ax.axhline(y = mean, color = 'k', linestyle = '--')
        ax.axhline(y = hh, color = 'red', linestyle = '--')
        ax.axhline(y = ll, color = 'red', linestyle = '--')
    else:
        sns.distplot(y, bins = 30, color = 'green', ax = ax)
        ax.text(0.8, 0.9, 'HH: ' + str(hh), transform=ax.transAxes)
        ax.text(0.8, 0.8, 'LL: ' + str(ll), transform=ax.transAxes)
        ax.axvline(x = mean, color = 'k', linestyle = '--')
        ax.axvline(x = hh, color = 'red', linestyle = '--')
        ax.axvline(x = ll, color = 'red', linestyle = '--')            
    return ax


###############################################################################
# Alarm limit on time series
###############################################################################
def plot_alarm_limit_on_ts(i, mean, sd, df, ax):
    
    lower = mean - 3*sd
    upper = mean + 3*sd

    try:
        levels = df.index.levels[0]
        for level in levels:
            x = df.loc[level].iloc[:,0]
            y = df.loc[level].iloc[:,i].copy(deep = True)
            
            youtside = np.ma.masked_inside(y, lower, upper)
            yinside = np.ma.masked_outside(y, lower, upper)
            
            ax.plot(x, youtside, 'red', label = 'Abnormal')
            ax.plot(x, yinside, 'green', label = 'Normal')
    except:
        x = df.iloc[:,0]
        y = df.iloc[:,i].copy(deep = True)
        
        youtside = np.ma.masked_inside(y, lower, upper)
        yinside = np.ma.masked_outside(y, lower, upper)
        
        ax.plot(x, youtside, 'red', label = 'Abnormal')
        ax.plot(x, yinside, 'green', label = 'Normal')
         
    ax.axhline(y=lower, color = 'red', linestyle='--')
    ax.axhline(y=mean, color = 'k', linestyle='--')
    ax.axhline(y=upper, color = 'red', linestyle='--')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim(mean - 5*sd, mean + 5*sd)
    return ax


###############################################################################
# Simple plot of time series with color coded data subsetting 
###############################################################################
# Step 1: Get the data
tag_name = abnormal_df.columns[4]
loads = [90, 100]
margin = pd.Timedelta('5 days')
start_date = '2014-04-12 00:00:00'
end_date = '2015-12-07 00:00:00'
change_location_ind = get_load_change_index_locations_and_type(abnormal_df, loads)
valid_dates = get_all_start_end_time_within_specified_load_limit(change_location_ind, margin, abnormal_df)
load_indices = get_load_indices(valid_dates, abnormal_df)
date_indices = get_date_indices(start_date, end_date, abnormal_df)
indices = get_indices(load_indices, date_indices)

# Step 2: time series plotting
i = 1
tag_name = abnormal_df.columns[i]
f, ax = plt.subplots()
ax = plot_subset_by_boolean_index(i, indices, abnormal_df, ax, col = 'r', invert = True)
ax = plot_subset_by_boolean_index(i, indices, abnormal_df, ax, col = 'g', invert = False)
ax = set_y_limit(65, 110, ax)
ax = add_h_line(loads, ax)
ax = add_v_line([start_date, end_date], ax)
ax = rotate_x_label(ax, 90)
ax = add_y_label(tag_name, ax, col = 'k')

# Step 3: Zoomed in on time series plot with change point added
f, ax = plt.subplots()
ax = plot_subset_by_boolean_index(i, indices, abnormal_df, ax, col = 'r', invert = True)
ax = plot_subset_by_boolean_index(i, indices, abnormal_df, ax, col = 'g', invert = False)
ax = set_y_limit(65, 110, ax)
ax = add_h_line(loads, ax)
ax = add_v_line([start_date, end_date], ax)
ax = rotate_x_label(ax, 90)
ax = add_y_label(tag_name, ax, col = 'k')
ax = add_change_points(change_location_ind, abnormal_df, ax)
ax = set_x_limit('2014-10-16 00:00:00', '2014-11-27 00:00:00', ax)


# Step 4: Visualize the proper subsetting of the data
i = 1
tag_name = abnormal_df.columns[i]
f, ax = plt.subplots()
ax = plot_subset_by_boolean_index(i, indices, abnormal_df, ax, col = 'r', invert = True)
ax = plot_subset_by_boolean_index(i, indices, abnormal_df, ax, col = 'g', invert = False)
ax = set_y_limit(65, 110, ax)
ax = add_y_label(tag_name, ax, col = 'k')
ax = rotate_x_label(ax, 90)

i = 4
tag_name = abnormal_df.columns[i]
ax0 = create_twin_axis(ax)
ax0 = plot_subset_by_boolean_index(i, indices, abnormal_df, ax0, col = 'm', invert = True)
ax0 = plot_subset_by_boolean_index(i, indices, abnormal_df, ax0, col = 'c', invert = False)
ax0 = set_x_limit(start_date, end_date, ax0)
ax0 = set_y_limit(110, 130, ax0)
ax0 = add_y_label(tag_name, ax0, col = 'k')

# Step 3: histogram plotting
i = 4
tag_name = abnormal_df.columns[i]
x, y = get_df_subset(i, indices, abnormal_df)
mean, sd = get_mean_sd(y)
f, ax = plt.subplots()
ax = plot_histogram_with_alarm_limits(y, mean, sd, ax, vertical = False)
ax = add_plot_title(ax, tag_name)

# Step 4: histogram input data plotting
i = 4
f, ax = plt.subplots()
ax = plot_subset_by_boolean_index(i, indices, abnormal_df, ax, col = 'r', invert = True)
ax = plot_subset_by_boolean_index(i, indices, abnormal_df, ax, col = 'g', invert = False)
ax = set_x_limit(start_date, end_date, ax)
ax = set_y_limit(115, 122, ax)
ax = rotate_x_label(ax, 90)

# Step 5: histogram input and histogram plotting on the same graph
i = 4
tag_name = abnormal_df.columns[i]
x, y = get_df_subset(i, indices, abnormal_df)
mean, sd = get_mean_sd(y)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
ax1 = plot_subset_by_boolean_index(i, indices, abnormal_df, ax1, col = 'r', invert = True)
ax1 = plot_subset_by_boolean_index(i, indices, abnormal_df, ax1, col = 'g', invert = False)
ax1 = set_x_limit(start_date, end_date, ax1)
ax1 = set_y_limit(115, 122, ax1)
ax1 = rotate_x_label(ax1, 90)
ax1 = add_h_line([mean], ax1)
ax1 = add_y_label(tag_name, ax1, col = 'k')

ax2 = plot_histogram_with_alarm_limits(y, mean, sd, ax2)



# Step 5: alarm limit plotting on the time series
f, ax = plt.subplots()
ax = plot_alarm_limit_on_ts(i, mean, sd, abnormal_df, ax)
ax = add_y_label(tag_name, ax, col = 'k')


# Step 6: add histogram to time series plot 
i = 4
x, y = get_df_subset(i, indices, abnormal_df)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
plot_alarm_limit_on_ts(i, mean, sd, abnormal_df, ax1)
plot_histogram_with_alarm_limits(y, mean, sd, ax2)


        










