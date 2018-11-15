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

ind = get_numeric_index(raw_data, i = 3)
y = get_y(raw_data, ind, i = 2)
bins, p = get_pdf(y)
plt.plot(bins, p)


i = 1
ind = get_numeric_index(alarm_setting, i)
x = get_x(alarm_setting, ind)
y = get_y(alarm_setting, ind, i = 1)
len(x)
alarm_setting.shape
freq, edges = np.histogram(y.values, bins=20)
plt.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge")

for i in range(1,2):
    print(i)
    plot_ts(i, alarm_setting)
    plot_hist(i, alarm_setting)

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
h = col_2.values
mean = np.mean(h)
std = np.std(h)

cut_off = std*3
lower, upper = mean - cut_off, mean + cut_off

bins = np.arange(190, 225, 0.001)
y_1 = stats.norm.pdf(bins, mean, std)

fig, ax1 = plt.subplots()

ax1.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge", color = 'yellow')
#ax1.tick_params(axis='y', labelcolor=color)
#ax1.set_xticklabels(ax1.get_xticklabels(), rotation='vertical')
ax1.get_yaxis().set_visible(False)
ax2 = ax1.twinx()
ax2.plot(bins, y_1, color = 'green')
ax2.set_ylim(0,0.09)
ax2.axvline(x = mean, color = 'green')
ax2.axvline(x = lower, color = 'red')
ax2.axvline(x = upper, color = 'red')
ax2.get_yaxis().set_visible(False)
#color = 'tab:red'
#ax1.set_ylabel('Extraction Current Load', color=color)

  # instantiate a second axes that shares the same x-axis

#color = 'tab:blue'
#ax2.plot(time, col_2, color=color)
#ax2.tick_params(axis='y', labelcolor=color)

#color = 'tab:green'

#ax2.set_ylabel('T-8220 Press (Torr)', color=color)
#ax2.set_ylabel('E-8221A/B/C Inlet SM (kPa)', color=color)
#ax2.set_ylabel('E-8221A/B/C Inlet SM (kPa)', color=color)
#ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()






###############################################################################
# Alarm setting based on histogram
###############################################################################

plt.plot(bins, y_1)
plt.show()

freq, edges = np.histogram(h, bins=20)
plt.bar(edges[:-1], freq, width=np.diff(edges), ec="k", align="edge")



# Sample from a normal distribution using numpy's random number generator
samples = h
max_h = max(h)
min_h = min(h)
# Compute a histogram of the sample
bins = np.linspace(190, 225, 20)
histogram, bins = np.histogram(samples, bins=bins, normed=True)

bin_centers = 0.5*(bins[1:] + bins[:-1])

# Compute the PDF on the bin centers from scipy distribution object
pdf = stats.norm.pdf(bin_centers)

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, histogram, label="Histogram of samples")
#plt.plot(bin_centers, pdf, label="PDF")
plt.legend()
plt.show()




h.sort()
fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
plt.plot(h,fit,'-o')
plt.hist(h,normed=True)      #use this to draw histogram of your data
plt.show()

for item in h:
    if not np.isnan(item):
        print(item)



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
df = raw_data
col_num = 2
valid_ind = get_numeric_index(df, col_num)
time = get_x(df, valid_ind)
col_1 = df.iloc[valid_ind, 1]
col_2 = df.iloc[valid_ind, 2]
ma_2 = col_2.rolling(window = 24*60).mean()
col_3 = df.iloc[valid_ind, 3]
ma_3 = col_3.rolling(window = 24*60).mean()
col_4 = df.iloc[valid_ind, 4]
ma_4 = col_4.rolling(window = 24*60).mean()

#cross_cor = correlate(col_1[:3*24*60], col_2[:3*24*60], mode = 'same')
#plt.plot(cross_cor, 'o')
#t = 20*24*60
#ma = col_2[:t].rolling(window = 24*60).mean()
#
#plt.plot(col_1[:t], label = col_1.name)
#plt.plot(col_2[:t], label = col_2.name)
#plt.plot(ma, label = 'MA')
#plt.legend()


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_ylabel('Extraction Current Load', color=color)
ax1.plot(time, col_1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation='vertical')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#color = 'tab:blue'
#ax2.plot(time, col_2, color=color)
#ax2.tick_params(axis='y', labelcolor=color)

color = 'tab:green'
ax2.plot(time, ma_2, color=color)
ax2.set_ylabel('T-8220 Press (Torr)', color=color)
#ax2.set_ylabel('E-8221A/B/C Inlet SM (kPa)', color=color)
#ax2.set_ylabel('E-8221A/B/C Inlet SM (kPa)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()




pc_change = col_1.pct_change(1)
plt.plot(pc_change*10000)
plt.ylim([0,100])
plt.plot(col_1)

pc_change = col_2.pct_change(1)
plt.plot(pc_change*10000)
plt.ylim([0,250])
plt.plot(col_2)

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












