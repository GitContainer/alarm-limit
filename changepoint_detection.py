# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:06:12 2018

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
# Column description
###############################################################################
def get_col_desc():
    col_desc = ['EXTRACTION CURRENT LOAD',	'T-8220 PRESS.',
  'E-8221A/B/C INLET SM', 'T-8220 BTM TEMP.',
  'E-8223 OUTLET LINE',	'E-8223 OUTLET LINE',
  'T-8220 BTM TEMP.', 'T-8220 BTM',	
  'E-8221A INLET SM', 'E-8221A/B/C SM',
   'E-8221A/B/C INLET OX',	'E-8221A/B INLET LINE']
    return col_desc
    
col_desc = get_col_desc()

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
# clean the data (Remove the non numeric characters)
###############################################################################
def remove_non_numeric_values(df):
    ind = []
    for j, n in enumerate(df.iloc[:,1]):
        if np.isreal(n):
            if not np.isnan(n):
                ind.append(j)
    cleaned_df = df.loc[ind,:]
    return cleaned_df
    
cleaned_df = remove_non_numeric_values(plug_data)

###############################################################################
#  Subset the data with load above certain threshold and within a given period 
###############################################################################    

def get_index_with_more_than_specified_load(df, load):
    ind = df.iloc[:,1] > load
    return ind
    

def get_change_locations(ind):
    change_locations = []
    change_values = []
    indices = ind.index
    values = ind.values
    n = len(indices)
    
    for i, val in enumerate(values):
        if i == n-1:
            break
        if val != values[i+1]:
            print(i, indices[i], val, values[i+1])
            change_locations.append(indices[i])
            change_values.append(val)
    return change_locations, change_values


def get_index_to_be_deleted(margin, change_locations, change_values):
    del_index = []    
    n = len(change_locations)
    for i, xc in enumerate(change_locations):
        if i == n-1:
            break
        if i == 0:
            continue
        pre_ind = change_locations[i - 1]
        ind = change_locations[i]
        
        if change_values[i] == True:
            time_diff = cleaned_df.iloc[ind, 0] - cleaned_df.iloc[pre_ind, 0]
            print(i, i-1, time_diff)
            print(i, cleaned_df.iloc[pre_ind,0], cleaned_df.iloc[ind, 0], change_values[i])
            if time_diff < margin:
                del_index.append(i-1)
                del_index.append(i)
    return del_index
        
def update_change_locations(change_locations, change_values, del_index):
    updated_locations = []
    updated_values = []
    
    for i, xc in enumerate(change_locations):
        if i in del_index:
            continue
        updated_locations.append(change_locations[i])
        updated_values.append(change_values[i])
    return updated_locations, updated_values


def plot_demarcation(df, updated_locations, updated_values, margin):
    plt.plot(df.iloc[:,0], df.iloc[:,1])
    for xc, val in zip(updated_locations, updated_values):
        if val == True:
            plt.axvline(x = df.iloc[xc,0] - margin, color = 'red',
                        linestyle = '--', linewidth = 0.5)
        elif val == False:
            plt.axvline(x = df.iloc[xc,0] + margin, color = 'green', 
                        linestyle = '--', linewidth = 0.5)
    plt.xticks(rotation = 'vertical')  
    plt.show()
    return

cleaned_df = remove_non_numeric_values(plug_data)
ind = get_index_with_more_than_specified_load(cleaned_df, 90)
change_locations, change_values = get_change_locations(ind)
margin = pd.Timedelta('10 days')
del_index = get_index_to_be_deleted(margin, change_locations, change_values)
updated_locations, updated_values = update_change_locations(change_locations, change_values, del_index)
plot_demarcation(cleaned_df, updated_locations, updated_values, margin)

###############################################################################
# in this the margin is not taken care of
###############################################################################
def get_change_time_with_margin(updated_locations, updated_values, margin):
    change_time_with_margin = []
    for loc, status in zip(updated_locations, updated_values):
        if status == True:
            time = cleaned_df.iloc[loc,0] - margin
        else:
            time = cleaned_df.iloc[loc,0] + margin
        change_time_with_margin.append(time)
    return change_time_with_margin
    

def get_indices_for_each_region(change_time_with_margin, updated_values, df):
    indices_to_keep = []
        
    n = len(change_time_with_margin)
    
    for i, time in enumerate(change_time_with_margin):
        status = updated_values[i]
        if i == 0:
            if status == True:
                ind = df.iloc[:,0]  < time
                indices_to_keep.append(ind)
                
        if i == n-1:
            if status == False:
                ind = df.iloc[:,0]  > time
                indices_to_keep.append(ind)
            break
        
        if status == False:
            next_time = change_time_with_margin[i+1] 
            ind1 = df.iloc[:,0]  > time
            ind2 = df.iloc[:,0]  < next_time
            ind = ind1 & ind2
            indices_to_keep.append(ind)
    return indices_to_keep

def plot_all_regions(df, indices_to_keep, change_time_with_margin):
    plt.plot(df.iloc[:,0], df.iloc[:,1], color = 'red') 
    for ind in indices_to_keep:
        plt.plot(df.loc[ind].iloc[:,0], df.loc[ind].iloc[:,1], linewidth = 2,
                 color = 'green')
    for time in change_time_with_margin:
        plt.axvline(x = time, color = 'red',
                            linestyle = '--', linewidth = 0.5)
    plt.xticks(rotation = 'vertical')
    return


def show_accepted_and_rejected_regions_after_data_cleaning(df, indices_to_keep, change_time_with_margin):
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Extraction Current Load', color='blue')
    ax1.plot(df.iloc[:,0], df.iloc[:,1], color='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation='vertical')
    for ind in indices_to_keep:
        ax1.plot(df.loc[ind].iloc[:,0], df.loc[ind].iloc[:,1], linewidth = 2,
                 color = 'blue')
    for time in change_time_with_margin:
        ax1.axvline(x = time, color = 'red',
                            linestyle = '--', linewidth = 0.5)
    ax1.set_ylim(0,200)
    plt.xticks(rotation = 'vertical')
    
    
    ax2 = ax1.twinx()  
        
    y_label = df.columns[4]
    ax2.plot(df.iloc[:,0], df.iloc[:,4], color='red')
    ax2.set_ylabel(y_label, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    for ind in indices_to_keep:
        ax2.plot(df.loc[ind].iloc[:,0], df.loc[ind].iloc[:,4], linewidth = 2,
                 color = 'green')
    fig.tight_layout()     
    return

show_accepted_and_rejected_regions_after_data_cleaning(cleaned_df, indices_to_keep, change_time_with_margin)


change_time_with_margin = get_change_time_with_margin(updated_locations, updated_values, margin)
indices_to_keep = get_indices_for_each_region(change_time_with_margin, updated_values, cleaned_df)
plot_all_regions(cleaned_df, indices_to_keep, change_time_with_margin)
 

# check the column names
for i, col in enumerate(cleaned_df):
    print(i, col)
 
cleaned_df.columns[4]             
###############################################################################
# percentage change not working
###############################################################################
#ind = abs(df.iloc[:,1].pct_change()) < 20
#df = cleaned_df.loc[ind_time, :]
#plt.plot(df.iloc[:,0], df.iloc[:,1])
#plt.plot(df.loc[ind,:].iloc[:,0], df.loc[ind,:].iloc[:,1])
#plt.xticks(rotation = 'vertical')
#x = df.iloc[:,1].pct_change()
#plt.plot(x)
###############################################################################















