# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:21:39 2018

@author: Ravi Tiwari
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
###############################################################################
# items to be taken from changepoint_detection
# 1. cleaned_df
# 2. indices_to_keep
###############################################################################
cleaned_df.head()
indices_to_keep

i = 3  # desired tag
tag_name = cleaned_df.columns[i]
load_tag = cleaned_df.columns[1]
all_time = cleaned_df.iloc[:,0].values
all_tag_values = cleaned_df[tag_name].values
all_load_values = cleaned_df[load_tag].values

data1 = all_time
data2 = all_tag_values
data3 = all_load_values


def plot1(ax, data1, data2, data3, ind):
    ax.plot(data1, data2, color = 'gray')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    
    ax.plot(data1[ind], data2[ind], color = 'b')
    ax.tick_params(axis='y', labelcolor='b')
    ax0 = ax.twinx()
    ax0.plot(data1, data3, color = 'gray')
    for tick in ax0.get_xticklabels():
        tick.set_rotation(90)
    ax0.plot(data1[ind], data3[ind], color = 'r')
    ax0.tick_params(axis='y', labelcolor='r')
    return 

def plot2(ax, data1, data2, data3, ind):
    ax.plot(data1[ind], data2[ind], color = 'b')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.tick_params(axis='y', labelcolor='b')
#    ax.set_ylim(160,190)
    ax0 = ax.twinx()
    ax0.plot(data1[ind], data3[ind], color = 'r')
    ax0.tick_params(axis='y', labelcolor='r')
    ax0.set_ylim(50,120)
    return 


def plot3(ax, data, ind):
    value = data2[ind]
    x = np.array(value, dtype='float')
    sns.distplot(x, bins = 15, color = 'blue', ax=ax)
#    ax.set_xlim(160, 190)
    return 

for ind in indices_to_keep:   
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plot1(ax1, data1, data2, data3, ind)
    plot2(ax2, data1, data2, data3, ind)
    plot3(ax3, data2, ind)
    fig.tight_layout()


