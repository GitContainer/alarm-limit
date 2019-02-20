# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:03:38 2019

@author: Ravi Tiwari
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import xlrd

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
# read the file into excel
###############################################################################
def read_excel_file(folder, fname, skip_row, cols):
    full_path = os.path.join(folder, fname)
    xl = pd.ExcelFile(full_path)
    for sheet in xl.sheet_names:
        if 'Graphics' in sheet:
            continue
                    
        df = xl.parse(sheet, usecols = cols, skiprows = skip_row)
        yield(df)


def remove_duplicate_columns(df):
    colnames = df.columns.values
    new_colnames = []
    
    for col in colnames:
        new_colnames.append(col.upper())
        
    df.columns = new_colnames
    df = df.loc[:, ~df.columns.duplicated()]
    return df



def read_all_sheets(folder, fname, skip_row, cols):
    
    df_seq = read_excel_file(folder, fname, skip_row, cols)
    
    for df in df_seq:
        df = remove_duplicate_columns(df)
        try:
            df_final = df_final.append(df)
        except:
            df_final = df
    return df_final

    
def create_date_index(df):
    df = df.rename(columns = {df.columns[0]: "datetime"})
    df = df.set_index('datetime')
    return df

def remove_non_numeric_data(df):
    _, n = df.shape
    for i in range(n):
        df.iloc[:,i] = pd.to_numeric(df.iloc[:,i], errors='coerce')
    
    df = df.dropna(axis = 0, how = 'any')
    return df

def get_cleaned_df(folder, fname, skip_row, cols):
    df = read_all_sheets(folder, fname, skip_row, cols)
    df = create_date_index(df)
    df = remove_non_numeric_data(df)
    return df
    
                    
###############################################################################
# Reading and saving files one by one
###############################################################################
def print_tank_name_and_associated_files(cat_dict):
    for key in cat_dict.keys():
        if 'T' in key:
            print(key)
            print(cat_dict[key])
            print('\n')

print_tank_name_and_associated_files(cat_dict1)

###############################################################################
# Opening and loading files for a given tank
###############################################################################

def load_files_for_a_given_tank(tank_name, skip_row, cols, cat_dict, folder):
    df = []
    for i, fname in enumerate(cat_dict1[tank_name]):
        if 'Plugging.xlsx' in fname:
            continue
        print(fname)
        open_a_single_file(fname, folder)
        try:
            df_i = get_cleaned_df(folder, fname, skip_row[i], cols[i])
        except:
            df_i = get_cleaned_df(folder, fname, skip_row, cols)
        
        df.append(df_i)
    return df
    

###############################################################################
# Tank 'T-1330'
###############################################################################
folder = 'D:\\sumitomo\\data'
tank_name = 'T-1330'
fname = cat_dict1[tank_name][0]
skip_row = 1
cols = 'D:Q'

open_a_single_file(fname, folder)

t_1330 = load_files_for_a_given_tank(tank_name, skip_row, cols, cat_dict1, folder) 
 
# putting the data in a singel data frame
df_1330 = pd.concat(t_1330)

# pickle the file for easier retrieval
df_1330.to_pickle("./df_1330.pkl")  

# retrieve the pickled file
unpickled_df_1330 = pd.read_pickle("./df_1330.pkl")  
 


###############################################################################
# Tank 'T-8320'
###############################################################################
tank_name = 'T-8320'
fname = cat_dict1[tank_name][1]
skip_row = 1
cols = 'C:O'

open_a_single_file(fname, folder)
t_8320 = load_files_for_a_given_tank(tank_name, skip_row, cols, cat_dict1, folder) 
 
# putting the data in a single data frame
df_8320 = pd.concat(t_8320)

# pickle the file for easier retrieval
df_8320.to_pickle("./df_8320.pkl")  

# retrieve the pickled file
unpickled_df_8320 = pd.read_pickle("./df_8320.pkl")  
 
###############################################################################
# Tank 'T-8330'
###############################################################################
tank_name = 'T-8330'
fname = cat_dict1[tank_name][0]
skip_row = 1
cols = 'D:Q'

open_a_single_file(fname, folder)
t_8330 = load_files_for_a_given_tank(tank_name, skip_row, cols, cat_dict1, folder) 

 
# putting the data in a single data frame
df_8330 = pd.concat(t_8330)

# pickle the file for easier retrieval
df_8330.to_pickle("./df_8330.pkl")  

# retrieve the pickled file
unpickled_df_8330 = pd.read_pickle("./df_8330.pkl") 
  
###############################################################################
# Tanks 'T-1220'
###############################################################################
tank_name = 'T-1220'
skip_row = [2, 3]
cols = ['F:T', 'F:W']

cat_dict1[tank_name]
    
time_start = time.time()
t_1220 = load_files_for_a_given_tank(tank_name, skip_row, cols, cat_dict1, folder)
time_end = time.time()
time_elasped = time_end - time_start

# pickle the file for easier retrieval
t_1220[0].to_pickle("./df_1220_nr.pkl")
t_1220[1].to_pickle("./df_1220_pg.pkl")  

# retrieve the pickled file
df_1220_nr = pd.read_pickle("./df_1220_nr.pkl") 
df_1220_pg = pd.read_pickle("./df_1220_pg.pkl")
 
###############################################################################
# Tanks 'T-8220'
###############################################################################
tank_name = 'T-8220'
skip_row = [3, 3, 2]    # check these values
cols = ['F:Y', 'F:R', 'E:S']  # check these values

time_start = time.time()
t_8220 = load_files_for_a_given_tank(tank_name, skip_row, cols, cat_dict1, folder)
time_end = time.time()
time_elasped = time_end - time_start


# pickle the file for easier retrieval
t_8220[0].to_pickle("./df_8220_pg.pkl")
t_8220[1].to_pickle("./df_8220_tm.pkl")  

# retrieve the pickled file
df_8220_pg = pd.read_pickle("./df_8220_pg.pkl") 
df_8220_tm = pd.read_pickle("./df_8220_tm.pkl")
        
###############################################################################
# Plot all variables in a data frame
###############################################################################
def plot_individual_variables(df):
    
    _, n = df.shape
    
    for i in range(n):
        df.plot(y = i)
        plt.show()
    return

plot_individual_variables(df_8220_pg)

###############################################################################
# reading individual line of an excel sheet
###############################################################################

def read_data_from_xl(fname, sheet_i, row = None, col = None):
    
    wb = xlrd.open_workbook(full_path)
    sheet = wb.sheet_by_index(sheet_i)
    
    if all([row, col]):
        return sheet.cell_value(row, col)
    if row:
        return sheet.row_values(row)
    if col:
        return sheet.col_values(col) 
    
    
folder = 'D:\\sumitomo\data'
tank_name = 'T-8220'
fname = cat_dict1[tank_name][0]
full_path = os.path.join(folder, fname)

col_names = read_data_from_xl(fname, 0, 2)
print(col_names)

###############################################################################
# Open the graphics file
###############################################################################
def open_graphics_file(folder):
    fnames = os.listdir(folder)    
    fname = [s for s in fnames if 'graphics' in s]    
    full_path = os.path.join(folder, fname[0])
    os.startfile(full_path)
    return

open_graphics_file(folder)



   
















