# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:03:38 2019

@author: Ravi Tiwari
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Set up the working directory
###############################################################################
os.chdir('D:\\sumitomo\\code\\alarm-limit')
os.listdir('.')


###############################################################################
# function to get file size in human readable format
###############################################################################
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


###############################################################################
# Organizing the files based on equipment
###############################################################################
def get_file_list_in_a_folder(folder):
    fnames = os.listdir(folder)
    return fnames


def organize_the_file_names_in_cat(fnames):
    cat_dict = {}
    for fname in fnames:
        cat = fname.split()[1]
        try:
            cat_dict[cat].append(fname)
        except:
            cat_dict[cat] = []
            cat_dict[cat].append(fname)
    return cat_dict
        
###############################################################################
# get the file sizes
###############################################################################
def print_file_sizes(cat_dict, folder):
    for fnames in cat_dict.values():
        print('\n')
        for fname in fnames:
            full_path = os.path.join(folder, fname)
            st = os.stat(full_path)
            print(fname  +  ' : ' + sizeof_fmt(st.st_size))
    return
            



folder1 = 'D:\\sumitomo\data'
fnames1 = get_file_list_in_a_folder(folder1)

folder2 = 'D:\\sumitomo\\data\\New folder'
fnames2 = os.listdir(folder2)

cat_dict1 = organize_the_file_names_in_cat(fnames1)
cat_dict2 = organize_the_file_names_in_cat(fnames2) 

print_file_sizes(cat_dict1, folder1)
print_file_sizes(cat_dict2, folder2)    


###############################################################################
# open a single file
###############################################################################
def open_a_single_file(fname, folder):
    full_path = os.path.join(folder, fname)
    st = os.stat(full_path)    
    print(fname + ' : ' + sizeof_fmt(st.st_size))
    os.startfile(full_path)
    return

fname = 'SMM1 T-1330 temp data1.xlsx' 

open_a_single_file(fname, folder1)

###############################################################################
# Open all files related to a given equipment
###############################################################################

def open_all_files_of_a_tag(tag, cat_dict, folder):
    for fname in cat_dict[tag]:
        print(fname)
        full_path = os.path.join(folder, fname)
        os.startfile(full_path)
    return
    

folder1 = 'D:\\sumitomo\data'
tag = 'T-1330'
open_all_files_of_a_tag(tag, cat_dict1, folder1)

###############################################################################
# Open all files one by one for inspection
###############################################################################
def open_all_files_in_a_folder(folder):
    fnames = os.listdir(folder)
    for fname in fnames:
        full_path = os.path.join(folder, fname)
        st = os.stat(full_path)
        print(fname  +  ' : ' + sizeof_fmt(st.st_size))
        print('want to open: y or n')
        x = input()
        if x == 'y':            
            try:
                os.startfile(full_path)
            except:
                pass
    return

folder = 'D:\\sumitomo\data'
open_all_files_in_a_folder(folder)



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



   
















