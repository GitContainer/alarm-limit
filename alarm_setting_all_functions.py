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
# merge the three cleaned data frames
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

plot_ts(1, cleaned_df, col_desc)


###############################################################################
def get_change_point_location(df, load):
    ind = df.iloc[:,1] > load
    ind_int = ind*1
    change_point = ind_int.diff()
    change_ind = abs(change_point) == 1
    return change_point.loc[change_ind]
    
def plot_ts_with_change_point(i, df, desc, change_point):
    col_name = df.columns[i]
    for j in range(3):
        ind = change_point[j].index
        x_change = df.loc[j].loc[ind].iloc[:,0]
        x = cleaned_df.loc[j].iloc[:,0]
        y = cleaned_df.loc[j].iloc[:,i]
        plt.plot(x, y, color = 'blue') 
        for t, ct in zip(x_change, change_point[j]):
            if ct == 1:
                col = 'red'
            if ct == -1:
                col = 'green'
            plt.axvline(x = t, color = col, linewidth = 0.4)
    plt.xticks(rotation = 'vertical')
    plt.title(col_name + ' : ' + desc[i-1])    
    fname = 'results\\figures\\SMM' + col_name.replace('.', '') + '_ts' + '.jpeg'
#    plt.xlabel('Date (YYYY-MM)')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return

change_point_series = get_change_point_location(cleaned_df, 90)
plot_ts_with_change_point(1, cleaned_df, col_desc, change_point_series)

def get_index_to_delete(change_point, margin):
    ind_to_delete = []
    for j in range(3):
        for i, change_type in enumerate(change_point[j]):
        #    print(i, change_type)
            if change_type == 1:
                current_ind = change_point[j].index[i]
                current_time = df.loc[j].loc[current_ind].iloc[0]
                try:
                    next_ind = change_point[j].index[i + 1]
                except:
                    pass
                else:
                    next_time = df.loc[j].loc[next_ind].iloc[0]
                    diff = next_time - current_time
                    print(diff)
                    if diff < margin:
                        print('yes')
                        ind_to_delete.append((j, current_ind))
                        ind_to_delete.append((j, next_ind))
    return np.array(ind_to_delete)

def get_merged_change_point(change_point, ind_to_delete):
    merged_index = []
    for j in range(3):
        ind = ind_to_delete[:,0] == j
        merged_index.append(change_point_series[j].drop(index = ind_to_delete[ind,1]))
    
    merged_change_point = pd.concat(merged_index, keys = range(3))
    return merged_change_point 


margin = pd.Timedelta('10 days')                
ind_to_delete = get_index_to_delete(change_point_series, margin)
merged_change_point = get_merged_change_point(change_point_series, ind_to_delete)   
plot_ts_with_change_point(1, cleaned_df, col_desc, merged_change_point)

    












    
        
        
            


    
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

def plot_demarcation(df, updated_change_dict, margin):
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
plot_demarcation(cleaned_df, updated_change_dict, margin)
###############################################################################
# change indices
###############################################################################
def get_change_time_with_margin(updated_locations, updated_values, margin, df):
    change_time_with_margin = []
    for loc, status in zip(updated_locations, updated_values):
        if status == True:
            time = df.iloc[loc,0] - margin
        else:
            time = df.iloc[loc,0] + margin
        change_time_with_margin.append(time)
    return change_time_with_margin


def get_updated_change_time_dict(updated_change_dict):
    updated_change_time_dict = {}
    for key in updated_change_dict.keys():
        updated_locations = updated_change_dict[key]['change_locations']
        updated_values = updated_change_dict[key]['change_values']
        updated_change_time = get_change_time_with_margin(updated_locations, updated_values, margin, cleaned_df.loc[key])
        updated_change_time_dict[key] = updated_change_time
    return updated_change_time_dict


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

def get_indices_dict(updated_change_time_dict, updated_change_dict, df):
    updated_change_time_dict = get_updated_change_time_dict(updated_change_dict)
    indices_dict = {}
    for key in updated_change_time_dict.keys():
        change_time_with_margin = updated_change_time_dict[key]
        updated_values = updated_change_dict[key]['change_values']
        df_key = df.loc[key]
        indices_dict[key] = get_indices_for_each_region(change_time_with_margin, updated_values, df_key)
    return indices_dict
    
indices_dict =  get_indices_dict(updated_change_time_dict, updated_change_dict, cleaned_df)   

###############################################################################
# 
###############################################################################
def get_index_in_single_array(indices_to_keep):
    n = len(indices_to_keep[0])
    ind = np.zeros(n, dtype=bool)
    for int_ind in indices_to_keep:
        for i, val in enumerate(int_ind):
            if val == True:
                ind[i] = True
    return ind

def get_single_array_indices_dict(indices_dict):   
    single_array_indices_dict = {}    
    for key in indices_dict.keys():
        single_array_indices_dict[key] = get_index_in_single_array(indices_dict[key])
    return single_array_indices_dict 

single_array_indices_dict = get_single_array_indices_dict(indices_dict)
    
    

def plotting_clean_values(i, df, single_array_indices_dict):
    for key in single_array_indices_dict.keys():
        ind = single_array_indices_dict[key]
        x = df.loc[key].iloc[:,0]
        y = df.loc[key].iloc[:,i].copy(deep = True)
        inv_ind = np.invert(ind)
        y[inv_ind] = None
        plt.plot(x, y, '--g')
    plt.xticks(rotation = 'vertical')
    plt.show()
    return
    
plotting_clean_values(4, cleaned_df, single_array_indices_dict)


def plot_histogram_with_alarm_limits(i, df, single_array_indices_dict):
    y = []
    for key in single_array_indices_dict.keys():
        ind = single_array_indices_dict[key]
        y_int = df.loc[key].iloc[ind,i].values.copy()
        y_int = np.array(y_int, dtype='float') 
        y.extend(y_int)
    y =  np.array(y) 
    y.flatten()  
    mean = np.mean(y)
    sd = np.std(y)
    sns.distplot(y, bins = 30, color = 'green')
    plt.axvline(x = mean, color = 'k')
    plt.axvline(x = mean + 3*sd, color = 'red')
    plt.axvline(x = mean - 3*sd, color = 'red')
    plt.xlabel(df.columns[i])
    plt.show()
    return mean, sd

mean, sd = plot_histogram_with_alarm_limits(4, cleaned_df, single_array_indices_dict)


    
def plot_alarm_limit_on_ts(i, df, mean, sd):
    x = df.iloc[:,0]
    y = df.iloc[:,i].copy(deep=True)
    y = np.array(y, dtype='float')
    
    lower = mean - 3*sd
    upper = mean + 3*sd
    
    youtside = np.ma.masked_inside(y, lower, upper)
    yinside = np.ma.masked_outside(y, lower, upper)
    plt.plot(x, youtside, 'red', label = 'Abnormal')
    plt.plot(x, yinside, 'green', label = 'Normal')
     
    plt.axhline(y=lower, color = 'green', linestyle='--')
    plt.axhline(y=mean, color = 'k', linestyle='--')
    plt.axhline(y=upper, color = 'green', linestyle='--')
    plt.xticks(rotation = 'vertical')
    plt.ylim(mean - 20*sd, mean + 20*sd)
    plt.title(df.columns[i])  
    plt.legend()
    plt.show()
    return
    
plot_alarm_limit_on_ts(4, cleaned_df, mean, sd)


























plot_demarcation(cleaned_df, updated_locations, updated_values, margin)

















