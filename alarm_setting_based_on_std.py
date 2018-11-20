# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:49:49 2018

@author: Ravi Tiwari
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy.stats as stats

###############################################################################
# Data Preparation
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

###############################################################################
# Find out the reference region for the normal operation through visualization
###############################################################################
def plot_ts(i, start_time, end_time, df):
    ind = get_numeric_index(df, i)
    col_name = df.columns[i]
    x = get_x(df, ind)
    y = get_y(df, ind, i)
    plt.plot(x,y) 
    plt.axvline(x=start_time, color = 'green', linestyle='--')
    plt.axvline(x=end_time, color = 'green', linestyle='--')
    plt.xticks(rotation = 'vertical')
    plt.title(col_name)    
    fname = 'SMM_result\\' + col_name.replace('.', '') + '_ts' + '.jpeg'
    plt.xlabel('Date (YYYY-MM)')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return

def plot_ts_zoomed(i, start_time, end_time, df):    
    ind = get_numeric_index(df, i)
    col_name = df.columns[i]
    x = df.iloc[ind,0]
    y = get_y(df, ind, i)
    plt.plot(x,y, 'o') 
    plt.xticks(rotation = 'vertical')
    plt.title(col_name)
    plt.xlim(start_time, end_time) 
#    plt.xlabel('DD HH:MM')
    fname = 'SMM_result\\' + col_name.replace('.', '') + '_ts' + '.jpeg'
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return

###############################################################################
# Get the upper and lower limit of the alarm setting and plot it on the original 
# plot
###############################################################################
def get_alarm_settings(i, how_many_std, start_time, end_time, df):
    ind1 = df.iloc[:,0] > start_time  
    ind2 = df.iloc[:,0] < end_time 
    ind = ind1 & ind2
    df_subset = df.loc[ind, :]
    ind_num = get_numeric_index(df_subset, i)
    y = get_y(df_subset, ind_num, i)
    std = np.std(y)
    mean = np.mean(y)
    cut_off = std*how_many_std
    lower, upper = mean - cut_off, mean + cut_off
    return lower, upper, mean, std

def plot_ts_with_upper_lower_bound(i, lower, upper, df):
    ind = get_numeric_index(df, i)
    col_name = df.columns[i]
    x = get_x(df, ind)
    y = get_y(df, ind, i)
    youtside = np.ma.masked_inside(y, lower, upper)
    yinside = np.ma.masked_outside(y, lower, upper)
    plt.plot(x, youtside, 'red', label = 'Abnormal')
    plt.plot(x,yinside, 'green', label = 'Normal')
    plt.legend() 
    plt.axhline(y=lower, color = 'green', linestyle='--')
    plt.axhline(y=upper, color = 'green', linestyle='--')
    plt.xticks(rotation = 'vertical')
    plt.title(col_name)    
    fname = 'SMM_result\\' + col_name.replace('.', '') + '_ts' + '.jpeg'
    plt.xlabel('Date (YYYY-MM)')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return

###############################################################################
# get the data for plotting
###############################################################################
def get_df_subset_based_on_date(start_time, end_time, df):
    ind1 = df.iloc[:,0] > start_time  
    ind2 = df.iloc[:,0] < end_time 
    ind = ind1 & ind2
    df_subset = df.loc[ind, :]
    return df_subset

def get_pdf_data(i, df):
    ind_num = get_numeric_index(df, i)
    data = df.iloc[ind_num, i]     
    mean = np.mean(data)
    std = np.std(data)
    high = np.max(data)
    low = np.min(data)
    bins = np.arange(low - 4*std, high + 4*std, 0.001)
    p = stats.norm.pdf(bins, mean, std)
    return bins, p

def get_hist_data(i, df):
    ind_num = get_numeric_index(df, i)
    y = get_y(df, ind_num, i)
    freq, edges = np.histogram(y.values, bins=20)
    return freq, edges
          

###############################################################################
# Do the plotting
###############################################################################
def plot_alarm_setting_histogram(freq, edges, bins, p, mean, lower, upper, std, title): 
    fig, ax1 = plt.subplots()
    ax1.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge", color = 'yellow')
    ax1.get_yaxis().set_visible(False)
    ax1.set_xlim([mean - 4*std, mean + 4*std])
    ax2 = ax1.twinx()
    ax2.plot(bins, p)
    ax2.set_ylim(0,np.max(p)+0.01)
    ax2.axvline(x = mean, color = 'green')
    ax2.axvline(x = lower, color = 'red')
    ax2.axvline(x = upper, color = 'red')
    ax2.get_yaxis().set_visible(False)
    plt.title(title)
    fig.tight_layout()     
    plt.show()
    return

def plot_all(i, start_time, end_time, df):
    plot_ts(i, start_time, end_time, df)
    plot_ts_zoomed(i, start_time, end_time, df)
    lower, upper, mean, std = get_alarm_settings(i, 3, start_time, end_time, df)
    plot_ts_with_upper_lower_bound(i, lower, upper, df)
    df_subset = get_df_subset_based_on_date(start_time, end_time, df)
    bins, p = get_pdf_data(i, df_subset)
    freq, edges = get_hist_data(i, df_subset)
    title = df.columns[i]
    plot_alarm_setting_histogram(freq, edges, bins, p, mean, lower, upper, std, title)  
    return

###############################################################################
# Figure out normal operating region based on visual inspection
###############################################################################
def plot_data_from_raw_data(df):  
    start_time = '2017-06-01 00:00'
    end_time = '2017-11-10 00:00'
    for i in range(2,5):
        plot_all(i, start_time, end_time, df)
        
    start_time = '2017-06-18 00:00'
    end_time = '2017-06-23 00:00'
    for i in range(5,7):
        plot_all(i, start_time, end_time, df)

#    start_time = '2017-05-21 00:00'
#    end_time = '2017-05-26 00:00'
#    for i in range(5,7):
#        plot_all(i, start_time, end_time, df)
    return        

plot_data_from_raw_data(df = raw_data)
    
###############################################################################
# case 1
###############################################################################
def plot_data_from_alarm_setting(df):  
    start_time = '2016-08-22 00:00'
    end_time = '2017-01-10 00:00'
    i = 1
    plot_all(i, start_time, end_time, df)
    return

plot_data_from_alarm_setting(df = alarm_setting)
###############################################################################
# case 2
###############################################################################











