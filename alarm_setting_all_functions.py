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
def read_xl_data(i, name):    
    col_no = range(5,18)
    if name == 'plug_data':
        fname = 'data\\IOT\\shared on 13.12.18\\SMM1 T-1220 Plugging.xlsx'
    if name == 'abnormality_data':
        fname = 'data\\IOT\\shared on 13.12.18\\Abnormality Detection May17 _ Jan18.xlsx'
        
    df = pd.read_excel(fname, usecols = col_no, skiprows = 3, sheet_name=i)    
    return df


###############################################################################
# clean the data (Remove the non numeric characters)
###############################################################################
#def remove_non_numeric_values(df):
#    ind = []
#    for j, n in enumerate(df.iloc[:,1]):
#        if np.isreal(n):
#            if not np.isnan(n):
#                ind.append(j)
#    cleaned_df = df.loc[ind,:]
#    return cleaned_df

def remove_non_numeric_values(df):
    n, _ = df.shape
    ind = []
    for i in range(n):
        row_vals = df.iloc[i,:].values
        ind_list = [np.isreal(x) for x in row_vals]
        ind.append(all(ind_list))
    cleaned_df = df.loc[ind,:]
    return cleaned_df

###############################################################################
# column description
###############################################################################
def get_col_desc(name):
    if name == 'abnormality_data':
        col_desc = ['EXTRACTION CURRENT LOAD',	'T-8220 PRESS.',
                    'E-8221A/B/C INLET SM', 'T-8220 BTM TEMP.',
                    'E-8223 OUTLET LINE',	'E-8223 OUTLET LINE',
                    'T-8220 BTM TEMP.', 'T-8220 BTM',	
                    'E-8221A INLET SM', 'E-8221A/B/C SM',
                    'E-8221A/B/C INLET OX',	'E-8221A/B INLET LINE']
        
    if name == 'plug_data' :
        col_desc = ['MAA EXTRACTION LOAD', 'T-1220', 'E-1221A/B INLET LINE', 
                    'T-1220 BOTTOM LINE',	'E-1223 OUTLET LINE', 'E-1223 OUTLET LINE',
                    'T-1220 BOTTOM LINE',	'T-1220 BOTTOM', 'E-1221A STEAM', 
                    'E-1221A/B INLET LINE', 'E-1221A/B INLET LINE', 'E-1221A/B  INLET LINE']
    return col_desc
    

###############################################################################
# data visualization
###############################################################################
def plot_ts(i, df, desc):
    col_name = df.columns[i]
    for j in range(3):
        x = df.loc[j].iloc[:,0]
        y = df.loc[j].iloc[:,i]
        plt.plot(x, y, color = 'blue') 
    plt.xticks(rotation = 'vertical')
    plt.title(col_name + ' : ' + desc[i-1])    
    fname = 'results\\figures\\SMM' + col_name.replace('.', '') + '_ts' + '.jpeg'
#    plt.xlabel('Date (YYYY-MM)')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return


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
        x = df.loc[j].iloc[:,0]
        y = df.loc[j].iloc[:,i]
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

def get_index_to_delete(change_point, margin, df):
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
                    if diff < margin:
                        ind_to_delete.append((j, current_ind))
                        ind_to_delete.append((j, next_ind))
    return np.array(ind_to_delete)

def get_merged_change_point(change_point, ind_to_delete):
    merged_index = []
    for j in range(3):
        ind = ind_to_delete[:,0] == j
        merged_index.append(change_point[j].drop(index = ind_to_delete[ind,1]))
    
    merged_change_point = pd.concat(merged_index, keys = range(3))
    return merged_change_point 

def get_index_for_specified_load(j, merged_change_point, df, margin):
    start_date = df.loc[j].iloc[0,0]
    end_date = df.loc[j].iloc[-1,0]
    n, _ = df.loc[j].shape
    ind = np.zeros(n, dtype = bool)
    for i, item in enumerate(merged_change_point[j]):
        if item == 1:
            st_ind = merged_change_point[j].index[i]
            start_date = df.loc[j].iloc[st_ind,0]
            try:
                en_ind = merged_change_point[j].index[i+1]
            except:
                end_date = df.loc[j].iloc[-1,0]
            else:
                end_date = df.loc[j].iloc[en_ind,0]
            ind1 = df.loc[j].iloc[:,0]  > start_date + margin
            ind2 = df.loc[j].iloc[:,0]  < end_date - margin
            c_ind = ind1 & ind2
            ind = ind | c_ind
    return ind

def get_merged_indices(merged_change_point, df, margin = pd.Timedelta('2 days')):
    merged_ind = []
    for j in range(3):
        ind = get_index_for_specified_load(j, merged_change_point, df, margin)
        merged_ind.append(ind)
    merged_indices = pd.concat(merged_ind, keys = range(3))
    return merged_indices


def plot_values_in_specified_load_and_margin(i, merged_indices, df):
    title = df.columns[i]
    for j in range(3):
        ind = merged_indices[j]
        x = df.loc[j].iloc[:,0] 
        y = df.loc[j].iloc[:,i].copy(deep = True)
        inv_ind = np.invert(ind)
        y[inv_ind] = None
        plt.plot(x,y)
    plt.title(title)    
    plt.xticks(rotation = 'vertical')
    plt.show()
    return
    
###############################################################################
# histogram with alarm limits
###############################################################################
def plot_histogram_with_alarm_limits(i, merged_indices, df):
    y = []
    for j in range(3):
        ind = merged_indices[j] 
        y_int = df.loc[j].loc[ind].iloc[:,i].values.copy()
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
    plt.ylim(mean - 10*sd, mean + 10*sd)
    plt.title(df.columns[i])  
    plt.legend()
    plt.show()
    return

   
###############################################################################
# moving average
###############################################################################
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_moving_average(i, df, window_size = 1*24*60):
    for j in range(3):
        x = df.loc[j].iloc[:,0]
        y = df.loc[j].iloc[:,i].copy(deep=True)
        y = np.array(y, dtype='float')
        y_mv = moving_average(y, n=window_size)
        plt.plot(x,y, color = 'blue')
        plt.plot(x[window_size-1:], y_mv, color = 'red')
    plt.xticks(rotation = 'vertical')
    plt.show()
    return

###############################################################################
# Final Run: Plugged Data
###############################################################################
# 1. Read
plug_data_0 = read_xl_data(0, 'plug_data')
plug_data_1 = read_xl_data(1, 'plug_data')
plug_data_2 = read_xl_data(2, 'plug_data')

# 2. Clean
cleaned_plug_df_0 = remove_non_numeric_values(plug_data_0)
cleaned_plug_df_1 = remove_non_numeric_values(plug_data_1)
cleaned_plug_df_2 = remove_non_numeric_values(plug_data_2)

# 3. Merge
frames = [cleaned_plug_df_2, cleaned_plug_df_0, cleaned_plug_df_1]
cleaned_plug_df = pd.concat(frames, keys=[0,1,2])

# 4. Column Description
plug_col_desc = get_col_desc('plug_data')

# 5. Visualize time series
i = 2
plot_ts(i, cleaned_plug_df, plug_col_desc)

# 6. Get and visualize change points
plug_change_points = get_change_point_location(cleaned_plug_df, 90)
plot_ts_with_change_point(1, cleaned_plug_df, plug_col_desc, plug_change_points)

# 7. merge nearby changepoints
margin = pd.Timedelta('10 days')                
ind_to_delete = get_index_to_delete(plug_change_points, margin, cleaned_plug_df)
plug_merged_change_point = get_merged_change_point(plug_change_points, ind_to_delete)   
plot_ts_with_change_point(1, cleaned_plug_df, plug_col_desc, plug_merged_change_point)
plug_merged_indices = get_merged_indices(plug_merged_change_point, cleaned_plug_df)

# 8. visualize the data within given load and with given margin 
plot_values_in_specified_load_and_margin(3, plug_merged_indices, cleaned_plug_df)

# 9. Get and plot alarm limit
mean, sd = plot_histogram_with_alarm_limits(i, plug_merged_indices, cleaned_plug_df)
plot_alarm_limit_on_ts(i, cleaned_plug_df, mean, sd)
plot_ts(i, cleaned_plug_df, plug_col_desc)

for i in range(2,13):
    mean, sd = plot_histogram_with_alarm_limits(i, plug_merged_indices, cleaned_plug_df) 

# 10. plot moving average
plot_moving_average(3, cleaned_plug_df, window_size = 5*24*60)

###############################################################################
# Final Run: Abnormality Data
###############################################################################
# 1. Read
abnormality_data_0 = read_xl_data(0, 'abnormality_data')
abnormality_data_1 = read_xl_data(1, 'abnormality_data')
abnormality_data_2 = read_xl_data(2, 'abnormality_data')

# 2. Clean
cleaned_abnormal_df_0 = remove_non_numeric_values(abnormality_data_0)
cleaned_abnormal_df_1 = remove_non_numeric_values(abnormality_data_1)
cleaned_abnormal_df_2 = remove_non_numeric_values(abnormality_data_2)

# 3. Merge
frames = [cleaned_abnormal_df_1, cleaned_abnormal_df_0, cleaned_abnormal_df_2]
cleaned_abnormal_df = pd.concat(frames, keys=[0,1,2])

# 4. Abnormal column description
abnormal_col_desc = get_col_desc('abnormality_data')

# 5. Visualize time series
i = 2
plot_ts(i, cleaned_abnormal_df, abnormal_col_desc)

# 6. Get and visualize change points
abnormal_change_points = get_change_point_location(cleaned_abnormal_df, 90)
plot_ts_with_change_point(1, cleaned_abnormal_df, abnormal_col_desc, abnormal_change_points)

# 7. merge nearby changepoints
margin = pd.Timedelta('10 days')                
ind_to_delete = get_index_to_delete(abnormal_change_points, margin, cleaned_abnormal_df)
abnormal_merged_change_point = get_merged_change_point(abnormal_change_points, ind_to_delete)   
plot_ts_with_change_point(1, cleaned_abnormal_df, abnormal_col_desc, abnormal_merged_change_point)
abnormal_merged_indices = get_merged_indices(abnormal_merged_change_point, cleaned_abnormal_df)


# 8. visualize the data within given load and with given margin 
plot_values_in_specified_load_and_margin(5, abnormal_merged_indices, cleaned_abnormal_df)

# 9. Get and plot alarm limit
mean, sd = plot_histogram_with_alarm_limits(i, abnormal_merged_indices, cleaned_abnormal_df)
plot_alarm_limit_on_ts(i, cleaned_abnormal_df, mean, sd)
plot_ts(i, cleaned_abnormal_df, abnormal_col_desc)

for i in range(2,13):
    mean, sd = plot_histogram_with_alarm_limits(i, abnormal_merged_indices, cleaned_abnormal_df) 
    plot_alarm_limit_on_ts(i, cleaned_abnormal_df, mean, sd)
    plot_ts(i, cleaned_abnormal_df, abnormal_col_desc)

# 10. plot moving average
for i in range(1,13):
    plot_moving_average(i, cleaned_abnormal_df, window_size = 5*24*60)
    

###############################################################################
# calculation for alarm setting
###############################################################################
def read_alarm_setting_file_old():    
    fname = 'data\\IOT\\shared on 13.12.18\\Alarm Settings HH PH PL LL.xlsx'
    col_no = range(6,8)
    alarm_setting = pd.read_excel(fname, usecols = col_no, skiprows = 2, sheet_name = 0)
    return alarm_setting    


def read_alarm_setting_file():    
    fname = 'data\\IOT\\shared on 13.12.18\\Alarm Settings HH PH PL LL.xlsx'
    col_no = range(4,10)
    alarm_setting = pd.read_excel(fname, usecols = col_no, skiprows = 3, sheet_name = 1)
    return alarm_setting    

alarm_setting_old = read_alarm_setting_file_old()    
alarm_setting = read_alarm_setting_file()

# plot alarm setting old data 
cleaned_aso = remove_non_numeric_values(alarm_setting_old)
x = cleaned_aso.iloc[:,0]
y = cleaned_aso.iloc[:,1]   
plt.plot(x, y)
plt.xticks(rotation = 'vertical')    
    
# plot alarm setting new data
cleaned_as = remove_non_numeric_values(alarm_setting)
for i in range(1,6): 
    title = cleaned_as.columns[i]
    x = cleaned_as.iloc[:,0]
    y = cleaned_as.iloc[:,i]   
    plt.plot(x, y)
    plt.xticks(rotation = 'vertical') 
    plt.title(title)
    plt.show()