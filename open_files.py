# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:03:38 2019

@author: Ravi Tiwari
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# 2.workstation
os.chdir('D:\\sumitomo\\code\\alarm-limit')
os.listdir('.')



# function to get file size in human readable format
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# get all file name
folder = 'D:\\sumitomo\data'
fnames = os.listdir(folder)

for fname in fnames:
    if fname == 'New folder':
        pass
    else:
        print(fname)
        full_path = os.path.join(folder, fname)
        st = os.stat(full_path)
        print('   :' + sizeof_fmt(st.st_size))
        

###############################################################################
# open a file
###############################################################################
full_path = os.path.join(folder, fnames[2])
st = os.stat(full_path)
sizeof_fmt(st.st_size)
os.startfile(full_path)
        
###############################################################################
 

###############################################################################
# open files one by one
###############################################################################
def open_file_one_by_one(fnames, folder):
    for fname in fnames:
        full_path = os.path.join(folder, fname)
        try:
            os.startfile(full_path)
            yield
        except:
            pass
        
# get the list of all the files
folder = 'D:\\sumitomo\data'
fnames = os.listdir(folder)
                
fopen = open_file_one_by_one(fnames, folder)

try:
    next(fopen)
except:
    pass


###############################################################################
# reading the files one by one
###############################################################################
folder = 'D:\\sumitomo\data'
fname = 'SMM1 T-1330 temp data1.xlsx'
full_path = os.path.join(folder, fname)
os.startfile(full_path)

xl = pd.ExcelFile(full_path)
xl.sheet_names 
df = xl.parse('Sheet1')
df.columns

for column in df.columns:
    print(df[column].head())
    
    
# find all the unnamed columns in the data

for column in df.columns:
    if 'Unnamed' in column:
        start_col = column

df = df.loc[:,start_col:]

df.tail()

# removing the empty rows
ind = df.iloc[:,0] == ' '
df = df.loc[~ind,:]

# creating date index
df_di = df.rename(columns = {"Unnamed: 3": "datetime"})
df_di = df_di.sort_values(by = 'datetime')
df_di = df_di.set_index('datetime')


# plot to see the variables
_, n = df_di.shape
for i in range(n):
    df_di.plot(y = i)
    plt.show()
    

###############################################################################
# reading and visualizing the files programmatically
###############################################################################
# opening the graphics files

folder = 'D:\\sumitomo\data'
fnames = os.listdir(folder)

fname = [s for s in fnames if 'graphics' in s]

full_path = os.path.join(folder, fname[0])
os.startfile(full_path)


###############################################################################
# next organizing the data (equipment wise)
###############################################################################
folder1 = 'D:\\sumitomo\data'
fnames1 = os.listdir(folder)

folder2 = 'D:\\sumitomo\\data\\New folder'
fnames2 = os.listdir(folder)

cat_dict = {}
for fname in fnames1:
    cat = fname.split()[1]
    try:
        cat_dict[cat].append(fname)
    except:
        cat_dict[cat] = []
        cat_dict[cat].append(fname)
        
for key in cat_dict.keys():
    print(key)
    print(cat_dict[key])        

cat_dict['T-1220']   
 

    
   

fnames1














