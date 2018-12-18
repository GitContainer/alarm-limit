# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:52:57 2018

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
def read_plug_data(i):    
    col_no = range(5,18)
    plug_data = pd.read_excel('data\\IOT\\shared on 13.12.18\\SMM1 T-1220 Plugging.xlsx',
                                 usecols = col_no, skiprows = 3, sheet_name=i)
    return plug_data

plug_data_0 = read_plug_data(0)
plug_data_1 = read_plug_data(1)
plug_data_2 = read_plug_data(2)



###############################################################################
# merge the three data frames
###############################################################################
frames = [plug_data_0, plug_data_1, plug_data_2]
plug_data = pd.concat(frames)
###############################################################################


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
    
cleaned_df_0 = remove_non_numeric_values(plug_data_0)
cleaned_df_1 = remove_non_numeric_values(plug_data_1)
cleaned_df_2 = remove_non_numeric_values(plug_data_2)
###############################################################################

###############################################################################
# check if all the three df have the same shape and same column names
###############################################################################
plug_data_0.shape, plug_data_1.shape, plug_data_2.shape
cleaned_df_0.shape, cleaned_df_1.shape, cleaned_df_2.shape
cleaned_df_0.columns
cleaned_df_1.columns
cleaned_df_2.columns

###############################################################################
# merge the three data frames
###############################################################################
frames = [cleaned_df_2, cleaned_df_0, cleaned_df_1]
cleaned_df = pd.concat(frames, keys=[0,1,2])
###############################################################################
# column description
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
# data visualization
###############################################################################
def plot_ts(i, df, desc):
    col_name = df.columns[i]
    for j in range(3):
        x = cleaned_df.loc[j].iloc[:,0]
        y = cleaned_df.loc[j].iloc[:,i]
        plt.plot(x, y, color = 'blue') 
    plt.xticks(rotation = 'vertical')
    plt.title(col_name + ' : ' + desc[i-1])    
    fname = 'results\\figures\\SMM' + col_name.replace('.', '') + '_ts' + '.jpeg'
#    plt.xlabel('Date (YYYY-MM)')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return

plot_ts(4, cleaned_df, col_desc)


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

def get_change_dictionary(ind):    
    change_dict = {}
    for i in range(3):
        change_locations, change_values = get_change_locations(ind[i])
        change_dict[i] = {}
        change_dict[i]['change_locations'] = change_locations
        change_dict[i]['change_values'] = change_values
    return change_dict
    
def get_index_to_be_deleted(margin, change_locations, change_values, df):
    '''if there are changes within that margin then those change locations are removed '''
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
            time_diff = df.iloc[ind, 0] - df.iloc[pre_ind, 0]
            print(i, i-1, time_diff)
            print(i, df.iloc[pre_ind,0], df.iloc[ind, 0], change_values[i])
            if time_diff < margin:
                del_index.append(i-1)
                del_index.append(i)
    return del_index

def get_del_dictionary(change_dict, margin, df):
    del_dict = {}
    for key in change_dict.keys():    
        change_locations = change_dict[key]['change_locations']
        change_values = change_dict[key]['change_values']
        df_i = df.loc[key]
        del_index = get_index_to_be_deleted(margin, change_locations, change_values, df_i)
        del_dict[key] = del_index
    return del_dict

def update_change_locations(change_locations, change_values, del_index):
    updated_locations = []
    updated_values = []
    
    for i, xc in enumerate(change_locations):
        if i in del_index:
            continue
        updated_locations.append(change_locations[i])
        updated_values.append(change_values[i])
    return updated_locations, updated_values

def get_updated_change_dict(change_dict, del_dict):
    updated_change_dict = {}
    for key in change_dict.keys():
        change_locations = change_dict[key]['change_locations']
        change_values = change_dict[key]['change_values']
        del_index = del_dict[key]
        updated_locations, updated_values = update_change_locations(change_locations, change_values, del_index)
        updated_change_dict[key] = {}
        updated_change_dict[key]['change_locations'] = updated_locations
        updated_change_dict[key]['change_values'] = updated_values
    return updated_change_dict

def plot_demarcation(df, updated_locations, updated_values, margin):
    for i in range(3):
        plt.plot(df.loc[i].iloc[:,0], df.loc[i].iloc[:,1], color = 'blue')
        updated_locations = updated_change_dict[i]['change_locations']
        updated_values = updated_change_dict[i]['change_values']
        for xc, val in zip(updated_locations, updated_values):
            if val == True:
                plt.axvline(x = df.loc[i].iloc[xc,0] - margin, color = 'red',
                            linestyle = '--', linewidth = 0.5)
            elif val == False:
                plt.axvline(x = df.loc[i].iloc[xc,0] + margin, color = 'green', 
                            linestyle = '--', linewidth = 0.5)
    plt.xticks(rotation = 'vertical')  
    plt.show() 
    return




ind = get_index_with_more_than_specified_load(cleaned_df, 90)    
change_dict = get_change_dictionary(ind)
margin = pd.Timedelta('10 days')
del_dict = get_del_dictionary(change_dict, margin, cleaned_df)
updated_change_dict = get_updated_change_dict(change_dict, del_dict)












plot_demarcation(cleaned_df, updated_locations, updated_values, margin)

















