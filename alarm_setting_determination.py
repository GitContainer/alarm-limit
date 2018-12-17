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
def plot1(ax, data1, data2, data3, ind):
    ax.plot(data1, data2, color = 'gray')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    
#    ax.plot(data1[ind], data2[ind], 'o', color = 'b', alpha = 0.001)
    ax.plot(data1[ind], data2[ind], linewidth = 0.2, color = 'b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_ylim(0,40)
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
#    ax0.set_ylim(50,120)
    return 


def plot3(ax, data, ind):
    value = data[ind]
    x = np.array(value, dtype='float')
    sns.distplot(x, bins = 10, color = 'blue', ax=ax)
#    ax.set_xlim(170, 280)
    return 


def get_data(df, i):
    tag_name = df.columns[i]
    load_tag = df.columns[1]
    all_time = df.iloc[:,0].values
    all_tag_values = df[tag_name].values
    all_load_values = df[load_tag].values
    return all_time, all_tag_values, all_load_values 



def create_plot_for_all_regions(data1, data2, data3, indices_to_keep, title):
    pre = 'results//SMM//T1210//' + title
    for i, ind in enumerate(indices_to_keep):   
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plot1(ax1, data1, data2, data3, ind)
        plot2(ax2, data1, data2, data3, ind)
        plot3(ax3, data2, ind)
        fig.suptitle(title)
        fig.tight_layout()
        ext = str(i) + '.png'
        fname = pre + ext
        fig.savefig(fname, bbox_inches='tight')
    return

def plot_tags_and_subregions(columns, df, indices_to_keep):
    for col in columns:
        data1, data2, data3 = get_data(df, col)
        title = df.columns[col]
        create_plot_for_all_regions(data1, data2, data3, indices_to_keep, title)
    return

plot_tags_and_subregions([11], cleaned_df, indices_to_keep)
        
for i, col in enumerate(cleaned_df.columns):
    print(i, col)       







