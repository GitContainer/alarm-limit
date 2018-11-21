# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 08:50:03 2018

@author: Ravi Tiwari
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy.stats as stats

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

def read_alarm_setting_file():    
    fname = 'IOT\\Alarm Settings HH PH PL LL.xlsx'
    col_no = range(6,8)
    alarm_setting = pd.read_excel(fname, usecols = col_no, skiprows = 2)
    return alarm_setting

raw_data = read_abnormality_data()
book = read_book1_data()
alarm_setting = read_alarm_setting_file()
    
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
# Plotting alarm setting
###############################################################################
def get_alarm_settings(i, how_many_std = 3, df = raw_data):
    ind = get_numeric_index(df, i)
    y = get_y(df, ind, i)
    std = np.std(y)
    mean = np.mean(y)
    cut_off = std*how_many_std
    lower, upper = mean - cut_off, mean + cut_off
    return lower, upper, mean, std

lower, upper, mean, std = get_alarm_settings(i = 2, how_many_std = 3, df = raw_data)

def get_pdf(data):    
    mean = np.mean(data)
    std = np.std(data)
    high = np.max(data)
    low = np.min(data)
    bins = np.arange(low - 10, high + 10, 0.001)
    p = stats.norm.pdf(bins, mean, std)
    return bins, p
    


def plot_alarm_setting_histogram(i, how_many_std = 3, df = raw_data):
    ind = get_numeric_index(df, i)
    y = get_y(df, ind, i)
    lower, upper, mean, std = get_alarm_settings(i = 2, how_many_std = 3, df = raw_data)
    freq, edges = np.histogram(y.values, bins=20)
    bins, p = get_pdf(y.values) 
    fig, ax1 = plt.subplots()
    ax1.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge", color = 'yellow')
    ax1.get_yaxis().set_visible(False)
    ax2 = ax1.twinx()
    ax2.plot(bins, p, color = 'green')
    ax2.set_ylim(0,np.max(p)+0.01)
    ax2.axvline(x = mean, color = 'green')
    ax2.axvline(x = lower, color = 'red')
    ax2.axvline(x = upper, color = 'red')
    ax2.get_yaxis().set_visible(False)
    fig.tight_layout()  
    plt.show()
    return


plot_alarm_setting_histogram(i = 2, how_many_std = 3, df = raw_data)    

def get_data_for_normal_region(df, i, date1, date2):
    ind1 = df.iloc[:,0] > date1  
    ind2 = df.iloc[:,0] < date2 
    ind = ind1 & ind2
    y = raw_data.iloc[:,i][ind]
    mean = np.mean(y)
    std = np.std(y)
    return mean, std

date1 = '2017-06-17 00:00'
date2 = '2017-06-23 00:00'
i = 5

mean, std = get_data_for_normal_region(raw_data, i, date1, date2)

mean
std

ind = get_numeric_index(raw_data, i)
x = raw_data.iloc[ind,0]
y = raw_data.iloc[ind,i]
signal = np.zeros(len(y))
signal_ind = np.zeros(len(y))
signal_ind = np.ones(len(y), dtype=bool)


threshold = 10
for i in range(len(y)):
    if abs(y.iloc[i] - mean) > threshold*std:
        signal[i] = 1
        signal_ind[i] = False


fig, ax1 = plt.subplots()
ax1.plot(x,y)
ax1.get_yaxis().set_visible(False)
ax2 = ax1.twinx()
ax2.plot(x, signal, color = 'green')
ax2.set_ylim(0,10)
fig.tight_layout()  
plt.show()


signal


plt.plot(x,y)
plt.plot(x, signal)

plt.plot(x[:10000], signal[:10000])
plt.plot(x[:10000], y[:10000])



i = 5
ind = get_numeric_index(raw_data, i)
x = get_x(raw_data, ind)


y = get_y(raw_data, ind, i )
lag = 50
threshold = 4
influence = 0

signals = np.zeros(len(y))
filteredY = y
avgFilter = np.zeros(len(y))
stdFilter = np.zeros(len(y))
avgFilter[lag] = np.mean(y[0:lag])
stdFilter[lag] = np.std(y[0:lag])

for i in range(lag+1, 5000):
    if abs(y[i] - avgFilter[i-1]) > threshold*stdFilter[i-1]:
        if y[i] > avgFilter[i-1]:
            signals[i] = 1
        else:
            signals[i] = -1
        filteredY[i] = influence*y[i]+(1-influence)*filteredY[i-1]
    else:
        signals[i] = 0
        filteredY[i] <- y[i]
    avgFilter[i] <- np.mean(filteredY[(i-lag):i])
    stdFilter[i] <- np.std(filteredY[(i-lag):i])
        
        

plt.plot(x[:5000],y[:5000])
plt.plot(x[:5000], signals[:5000])

plt.plot(y[:25000])


max(y)
min(y)
std = np.std(y)
mean = np.mean(y)
cut_off = std*3
lower, upper = mean - cut_off, mean + cut_off
outliers_removed = [x for x in y if x > lower and x < upper]
freq, edges = np.histogram(outliers_removed, bins=20)
plt.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge")

###############################################################################
# Alarm setting based on histogram
###############################################################################
#h = col_2.values
#mean = np.mean(h)
#std = np.std(h)
#
#cut_off = std*3
#lower, upper = mean - cut_off, mean + cut_off
#
#bins = np.arange(190, 225, 0.001)
#y_1 = stats.norm.pdf(bins, mean, std)
#
#fig, ax1 = plt.subplots()
#
#ax1.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge", color = 'yellow')
##ax1.tick_params(axis='y', labelcolor=color)
##ax1.set_xticklabels(ax1.get_xticklabels(), rotation='vertical')
#ax1.get_yaxis().set_visible(False)
#ax2 = ax1.twinx()
#ax2.plot(bins, y_1, color = 'green')
#ax2.set_ylim(0,0.09)
#ax2.axvline(x = mean, color = 'green')
#ax2.axvline(x = lower, color = 'red')
#ax2.axvline(x = upper, color = 'red')
#ax2.get_yaxis().set_visible(False)
##color = 'tab:red'
##ax1.set_ylabel('Extraction Current Load', color=color)
#
#  # instantiate a second axes that shares the same x-axis
#
##color = 'tab:blue'
##ax2.plot(time, col_2, color=color)
##ax2.tick_params(axis='y', labelcolor=color)
#
##color = 'tab:green'
#
##ax2.set_ylabel('T-8220 Press (Torr)', color=color)
##ax2.set_ylabel('E-8221A/B/C Inlet SM (kPa)', color=color)
##ax2.set_ylabel('E-8221A/B/C Inlet SM (kPa)', color=color)
##ax2.tick_params(axis='y', labelcolor=color)
#
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()






###############################################################################
# Alarm setting based on histogram
###############################################################################






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


plot_ts_zoomed(6, '2017-06-17 00:00', '2017-06-23 00:00', df = raw_data)
plot_ts_zoomed(3, '2017-08-04 11:10', '2017-08-04 11:30', df = raw_data)
plot_ts_zoomed(6, '2017-08-04 11:10', '2017-08-04 11:30', df = raw_data)
plot_ts_zoomed(5, '2017-08-04 11:10', '2017-08-04 11:30', df = raw_data)
    
raw_data.columns[0]
ind1 = raw_data.iloc[:,0] > '2017-06-17 00:00'  
ind2 = raw_data.iloc[:,0] < '2017-06-23 00:00'  
ind = ind1 & ind2
raw_data.iloc[:,0][ind]
raw_data.iloc[:,5][ind]

y = get_y(alarm_setting, ind, i = 1)




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
def plot_moving_average_and_load(i , mv_period = 24, df = raw_data):
    valid_ind = get_numeric_index(df, i)
    time = get_x(df, valid_ind)
    col_1 = df.iloc[valid_ind, 1]
    col_2 = df.iloc[valid_ind, i]
    ma_2 = col_2.rolling(window = mv_period*60).mean()
    title = df.columns[i]
    y_label = 'E-8221 A/B/C Inlet SM (kPa)'
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_ylabel('Extraction Current Load', color=color)
    ax1.plot(time, col_1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation='vertical')
    ax2 = ax1.twinx()  
        
    color = 'tab:green'
    ax2.plot(time, ma_2, color=color)
    ax2.set_ylabel(y_label, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(title)
    fig.tight_layout()  
    plt.show()
    return
    
plot_moving_average_and_load(i = 3 , mv_period = 10*24, df = raw_data)

def plot_moving_average(i, mv_period, df):
    valid_ind = get_numeric_index(df, i)
    time = get_x(df, valid_ind)
    col_2 = df.iloc[valid_ind, i]
    ma_2 = col_2.rolling(window = mv_period*60).mean()
    title = df.columns[i]
    
    plt.plot(time, col_2)
    plt.plot(time, ma_2, color = 'tab:green', label = 'Moving Average')
    plt.xticks(rotation = 'vertical')
    plt.title(title)
    plt.legend()
    plt.show()
    return

plot_moving_average(i = 3, mv_period = 10*24, df = raw_data)













###############################################################################
# Save to CSV
###############################################################################
col_2.to_csv('PI8221PV.txt')
col_1.to_csv('FC8215LDCPV.txt')


from scipy import signal
sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
sig_noise = sig + np.random.randn(len(sig))
corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128
plt.plot(sig_noise)
plt.plot(np.ones(128))
plt.plot(corr)
###############################################################################
# Moving average
###############################################################################
def plot_moving_average(df, col_num, window_length = 24):
    valid_ind = get_numeric_index(df, col_num)
    x = df.iloc[valid_ind, 0]
    subset = df.iloc[valid_ind, col_num]
    ma = subset.rolling(window=24*60).mean()
#    title = subset.name
    title = 'T-8220 Press (Torr)'
    fname = 'SMM_result\\' + title.replace('.', '') + '_ma' + '.jpeg'
    plt.plot(x, subset, color = 'yellow')
    plt.plot(x, ma, label = 'moving average (24hrs)', color = 'green')
    plt.xticks(rotation = 'vertical')
    plt.title(title)
    plt.xlabel('Date (YYYY-MM)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(fname)
    plt.show()
    return

for i in range(2,3):
    plot_moving_average(raw_data, i, window_length = 24)
    





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












