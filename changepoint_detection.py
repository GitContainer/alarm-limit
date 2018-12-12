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
# clean the data
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

   
    

def get_index_with_more_than_specified_load(df, load):
    ind = df.iloc[:,1] > load
    return ind
    
ind = get_index_with_more_than_specified_load(cleaned_df, 90)

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
        next_ind = change_locations[i + 1]
        
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
ind = cleaned_df.iloc[:,1] > 90
change_locations, change_values = get_change_locations(ind)
margin = pd.Timedelta('10 days')
del_index = get_index_to_be_deleted(margin, change_locations, change_values)
updated_locations, updated_values = update_change_locations(change_locations, change_values, del_index)
plot_demarcation(cleaned_df, updated_locations, updated_values, margin)

###############################################################################
# in this the margin is not taken care of
###############################################################################
ind = []
n_df, _ = cleaned_df.shape
n_locs = len(updated_locations)
for i, _ in enumerate(updated_locations):
    if i == 0:
        if updated_values[i] == True:
            print(updated_locations[i])
            locs = range(0,updated_locations[i])
            print(locs)
            ind.append(locs)
    elif i == (n_locs - 1):
        if updated_values[i] == False:
            print(updated_locations[i])
            locs = range(updated_locations[i], n_df)
            print(locs)
            ind.append(locs)
    else:
        if updated_values[i] == False:
            locs = range(updated_locations[i], updated_locations[i+ 1])
            print(locs)
            ind.append(locs)

           
for item in ind:
    plt.plot(cleaned_df.iloc[item,0], cleaned_df.iloc[item,1], linewidth = 2)
plt.xticks(rotation = 'vertical')
        
time_point = cleaned_df.iloc[203575,0] - margin 
ind = cleaned_df.iloc[:,0] < time_point

plt.plot(cleaned_df.iloc[:,0], cleaned_df.iloc[:,1]) 
plt.plot(cleaned_df.loc[ind].iloc[:,0], cleaned_df.loc[ind].iloc[:,1]) 
plt.xticks(rotation = 'vertical')    



for loc, val in zip(updated_locations, updated_values):
    print(loc, val, cleaned_df.iloc[loc,0])

plt.plot(cleaned_df.iloc[:200875,0], cleaned_df.iloc[:200875,1])
plt.plot(cleaned_df.iloc[203575:241924,0], cleaned_df.iloc[203575:241924,1])







###############################################################################
# plotting the cleaned data
###############################################################################
plt.plot(cleaned_df.iloc[:,0], cleaned_df.iloc[:,1])
plt.xticks(rotation = 'vertical')


ind = cleaned_df.iloc[:,1] > 90
df = cleaned_df.loc[ind, :]
plt.plot(cleaned_df.iloc[:,0], cleaned_df.iloc[:,1])
plt.plot(df.loc[ind,:].iloc[:,0], df.loc[ind,:].iloc[:,1])
plt.xticks(rotation = 'vertical')


start_time = '2017-01-09 11:27:00'
end_time = '2017-01-09 12:57:00'

ind1 = cleaned_df.iloc[:,0] > start_time
ind2 = cleaned_df.iloc[:,0] < end_time
ind_time = ind1 & ind2

df = cleaned_df.loc[ind_time, :]
plt.plot(df.iloc[:,0], df.iloc[:,1])
plt.xticks(rotation = 'vertical')


time_ind = ind == False
change_locations = []
change_values = []
indices = time_ind.index
values = time_ind.values
n = len(indices)

for i, val in enumerate(values):
    if i == n-1:
        break
    if val != values[i+1]:
        print(i, indices[i], val, values[i+1])
        change_locations.append(indices[i])
        change_values.append(val)


margin = pd.Timedelta('1 days')
start_time = '2017-11-24 11:27:00'
end_time = '2017-12-05 12:57:00'
    
plt.plot(cleaned_df.iloc[:,0], cleaned_df.iloc[:,1])
for xc, val in zip(change_locations, change_values):
    if val == True:
        plt.axvline(x = cleaned_df.iloc[xc,0] + margin, color = 'green',
                    linestyle = '--', linewidth = 0.5)
    elif val == False:
        plt.axvline(x = cleaned_df.iloc[xc,0] - margin, color = 'red', 
                    linestyle = '--', linewidth = 0.5)
plt.xticks(rotation = 'vertical')
plt.ylim(70,120)
plt.xlim(start_time, end_time)




for xc, val in zip(change_locations, change_values):
    print(cleaned_df.iloc[xc,0], val)


del_index = []

margin = pd.Timedelta('1 days')    
n = len(change_locations)
for i, xc in enumerate(change_locations):
    if i == n-1:
        break
    if i == 0:
        continue
    pre_ind = change_locations[i - 1]
    ind = change_locations[i]
    next_ind = change_locations[i + 1]
    
    if change_values[i] == False:
        time_diff = cleaned_df.iloc[ind, 0] - cleaned_df.iloc[pre_ind, 0]
        print(i, i-1, time_diff)
        print(i, cleaned_df.iloc[pre_ind,0], cleaned_df.iloc[ind, 0], change_values[i])
        if time_diff < margin:
            del_index.append(i-1)
            del_index.append(i)
#    if change_values[i] == True:
#        time_diff = cleaned_df.iloc[next_ind, 0] - cleaned_df.iloc[ind, 0]
#        print(i, i+1, time_diff)
#        print(i, cleaned_df.iloc[ind,0], cleaned_df.iloc[next_ind, 0], change_values[i])
#        if time_diff < margin:
#            del_index.append(i)
#            del_index.append(i+1)
        
updated_change_locations = []
updated_change_values = []
for i, xc in enumerate(change_locations):
    if i in del_index:
        continue
    updated_change_locations.append(change_locations[i])
    updated_change_values.append(change_values[i])

    
    
        
margin = pd.Timedelta('1 days')
start_time = '2017-11-24 11:27:00'
end_time = '2017-12-05 12:57:00'
    
plt.plot(cleaned_df.iloc[:,0], cleaned_df.iloc[:,1])
for xc, val in zip(updated_change_locations, updated_change_values):
    if val == True:
        plt.axvline(x = cleaned_df.iloc[xc,0] + margin, color = 'green',
                    linestyle = '--', linewidth = 0.5)
    elif val == False:
        plt.axvline(x = cleaned_df.iloc[xc,0] - margin, color = 'red', 
                    linestyle = '--', linewidth = 0.5)
plt.xticks(rotation = 'vertical')
plt.ylim(70, 120)
plt.xlim(start_time, end_time) 

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















