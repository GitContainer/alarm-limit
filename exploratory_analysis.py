# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 08:50:03 2018

@author: Ravi Tiwari
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
###############################################################################
# Set the working directory
###############################################################################
os.chdir('C:\\Users\\40204945\\Documents\\sumitomo')
os.listdir('.')

###############################################################################
# Read the data
###############################################################################
def read_abnormality_data():
    col_no = range(5,12)
    raw_data = pd.read_excel('IOT\\Abnormality Detection May17 ~ Jan18.xlsx',
                             usecols = col_no, skiprows = 3)
    return raw_data

def read_book1_data():
    col_no = range(6, 11)
    book1 = pd.read_excel('IOT\\Book1.xlsx', usecols = col_no, skiprows = 3)
    book1 = book1.drop(columns = ['Unnamed: 3'])
    
    col_no = range(12, 17)
    book2 = pd.read_excel('IOT\\Book1.xlsx', usecols = col_no, skiprows = 3)
    book2 = book2.drop(columns = ['Unnamed: 3'])
    
    book = pd.concat([book2, book1], axis = 0)
    return book

raw_data = read_abnormality_data()
book = read_book1_data()
    
###############################################################################
# Finding the distribution
###############################################################################
def get_numeric_index(df, i):
    ind = []
    for j, n in enumerate(df.iloc[:,i]):
        if np.isreal(n):
            if not np.isnan(n):
                ind.append(j)
    return ind

def get_x(df, ind):
    x = df.iloc[ind,0]
    return x
    
def get_y(df, ind, i = 2):
    y = df.iloc[ind,i]
    return y
    
def plot_ts(i, df = raw_data):
    ind = get_numeric_index(df, i)
    col_name = df.columns[i]
    x = get_x(df, ind)
    y = get_y(df, ind, i)
    plt.plot(x,y) 
    plt.xticks(rotation = 'vertical')
    plt.title(col_name)    
    fname = 'SMM_result\\' + col_name.replace('.', '') + '_ts' + '.jpeg'
    plt.xlabel('Date (YYYY-MM)')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return

def plot_hist(i, bins = 20, df = raw_data):
    ind = get_numeric_index(df, i)
    col_name = df.columns[i]
    y = get_y(df, ind, i)
    freq, edges = np.histogram(y.values, bins=bins)        
    plt.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge")
    plt.title(col_name)
    fname = 'SMM_result\\' + col_name.replace('.', '') + '_hist' + '.jpeg'
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    return
    
###############################################################################
# Plotting abnormality data
###############################################################################
def plot_abnormality_data(raw_data):   
    for i in range(1,7):
        plot_ts(i, raw_data)
        plot_hist(i, 20, raw_data)
    return

plot_abnormality_data(raw_data)
###############################################################################
# Plotting book data
###############################################################################
def plot_book1_data(book):    
    for i in range(1,4):
        plot_ts(i, book)
        plot_hist(i, 10, book)
    return
        
plot_book1_data(book)    
###############################################################################
# Zoom in to see clearly
###############################################################################
def plot_ts_zoomed(i, start_time, end_time, df = raw_data):    
    ind = get_numeric_index(raw_data, i)
    col_name = raw_data.columns[i]
    x = raw_data.iloc[ind,0]
    y = get_y(raw_data, ind, i)
    plt.plot(x,y, 'o') 
    plt.xticks(rotation = 'vertical')
    plt.title(col_name)
    plt.xlim(start_time, end_time) 
    plt.xlabel('DD HH:MM')
    fname = 'SMM_result\\' + col_name.replace('.', '') + '_ts' + '.jpeg'
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


plot_ts_zoomed(6, '2017-07-01 01:00', '2017-07-01 01:05', df = raw_data)
plot_ts_zoomed(3, '2017-08-04 11:10', '2017-08-04 11:30', df = raw_data)
plot_ts_zoomed(6, '2017-08-04 11:10', '2017-08-04 11:30', df = raw_data)
plot_ts_zoomed(5, '2017-08-04 11:10', '2017-08-04 11:30', df = raw_data)
    
###############################################################################
# Finding Correlation
###############################################################################
def get_correlation_matrix(df):
    ind = get_numeric_index(df, 1)
    data_matrix = df.iloc[ind,1:]
    data_matrix = data_matrix.values
    data_matrix = data_matrix.astype(float)
    cor_matrix = np.corrcoef(data_matrix.T)
    return cor_matrix
    
def get_colnames(df):
    colnames = list(df.columns.values)
    return colnames[1:]
    
def plot_cor_matrix(cormatrix, colnames):
    fname = 'SMM_result\\' + 'cormatrix.jpeg'
    fig, ax = plt.subplots()
    ax.imshow(cormatrix, cmap=plt.cm.Blues)
    
    ax.set_xticks(range(len(cormatrix)))
    ax.set_yticks(range(len(cormatrix)))
    ax.set_xticklabels(colnames)
    ax.set_yticklabels(colnames)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    for i in range(len(cormatrix)):
        for j in range(len(cormatrix)):
            ax.text(j, i, np.round(cormatrix[i, j], decimals = 2),
                           ha="center", va="center", color="w")
    
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    plt.savefig(fname)
    plt.show()
    return

cor_matrix = get_correlation_matrix(raw_data)
colnames = get_colnames(raw_data)
plot_cor_matrix(cor_matrix, colnames)


###############################################################################
# Cross correlation
###############################################################################




###############################################################################
# Moving average
###############################################################################
def plot_moving_average(df, col_num, window_length = 24):
    valid_ind = get_numeric_index(df, col_num)
    x = df.iloc[valid_ind, 0]
    subset = df.iloc[valid_ind, col_num]
    ma = subset.rolling(window=24*60).mean()
    
    plt.plot(x, subset)
    plt.plot(x, ma)
    plt.xticks(rotation = 'vertical')
    plt.show()
    return

plot_moving_average(raw_data, 2, window_length = 24)



###############################################################################
# might be useful in future
###############################################################################
#f, ax = plt.subplots(1, 2)
#ax[0].plot(x, y)
#
#ax[1].bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge")
#f, ax = plt.subplots(1, 2)
#ax[0].plot(x, y)
#ax[0].xticks(rotation = 'vertical')
#ax[1].bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge")
#
################################################################################
## plotting the data
################################################################################
#
#ind = []
#for i, n in enumerate(raw_data.iloc[:,2]):
#    if np.isreal(n):
#        ind.append(i)
#        
#x = raw_data.iloc[ind,0]
#
#for i in range(1,7):
#    print(i)
#    print(raw_data.columns[i])
#    y = raw_data.iloc[ind,i]
#    col_name = raw_data.columns[i]
#    plt.plot(x,y)
#    plt.xticks(rotation = 'vertical')
#    plt.title(col_name)
#    plt.show()
#    












