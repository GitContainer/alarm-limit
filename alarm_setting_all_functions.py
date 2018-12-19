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



for item in merged_change_point:
    print(item)    


ind_list = []
 





j = 0
start_date = cleaned_df.loc[j].iloc[0,0]
end_date = cleaned_df.loc[j].iloc[-1,0]
n, _ = cleaned_df.loc[j].shape
ind = np.zeros(n, dtype = bool)
for i, item in enumerate(merged_change_point[j]):
    if item == 1:
        st_ind = merged_change_point[j].index[i]
        start_date = cleaned_df.loc[j].iloc[st_ind,0]
        try:
            en_ind = merged_change_point[j].index[i+1]
        except:
            end_date = cleaned_df.loc[j].iloc[-1,0]
        else:
            end_date = cleaned_df.loc[j].iloc[en_ind,0]
        ind1 = cleaned_df.loc[j].iloc[:,0]  > start_date
        ind2 = cleaned_df.loc[j].iloc[:,0]  < end_date
        c_ind = ind1 & ind2
        ind = ind | c_ind
        
x = cleaned_df.loc[0].loc[ind,:].iloc[:,0]  
y = cleaned_df.loc[0].loc[ind,:].iloc[:,2]  
plt.plot(x,y)
plt.xticks(rotation = 'vertical')    
        

x = cleaned_df.loc[0].iloc[:,0] 
y = cleaned_df.loc[0].iloc[:,2].copy(deep = True)
inv_ind = np.invert(ind)
y[inv_ind] = None
plt.plot(x,y)
plt.xticks(rotation = 'vertical')   

        


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

















