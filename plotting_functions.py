# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:04:22 2018

@author: Ravi Tiwari
"""
import matplotlib.pyplot as plt


def plot_ts(i, df, desc, col = 'blue'):
    col_name = df.columns[i]
    f, ax = plt.subplots()
    for j in range(3):
        x = df.loc[j].iloc[:,0]
        y = df.loc[j].iloc[:,i]
        ax.plot(x, y, color = col) 
    plt.xticks(rotation = 'vertical')
    plt.title(col_name + ' : ' + desc[i-1])    
    return ax


def subset_ts_plot(start_date, end_date, i, df, desc, col = 'blue'):
    ax = plot_ts(1, df, desc, col)
    ax.set_xlim(start_date, end_date)
    return ax


start_date = '2014-04-16 00:00:00'
end_date = '2015-04-16 00:00:00'
i = 1

ax = plot_ts(i, cleaned_abnormal_df, abnormal_col_desc, col = 'red')
ax = subset_ts_plot(start_date, end_date, i, cleaned_abnormal_df, abnormal_col_desc, col = 'red')





abnormal_col_desc
cleaned_abnormal_df
abnormal_merged_indices

df = cleaned_abnormal_df

i = 1
