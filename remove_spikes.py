# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:31:56 2018

@author: Ravi Tiwari
"""

###############################################################################
# This module required the existence of the following two variables
# cleaned_df
# indices_to_keep
###############################################################################
def get_index_in_single_array(indices_to_keep):
    n = len(indices_to_keep[0])
    ind = np.zeros(n, dtype=bool)
    for int_ind in indices_to_keep:
        for i, val in enumerate(int_ind):
            if val == True:
                ind[i] = True
    return ind

def plotting_clean_values(i, df, ind):
    title = df.columns[i]
    x = df.iloc[:,0]
    y = df.iloc[:,i].copy(deep = True)
    inv_ind = np.invert(ind)
    y[inv_ind] = None
    
    plt.plot(x, y, '--g')
    plt.xticks(rotation = 'vertical')
    plt.title(title)
    plt.show()
    return

def plotting_mv_clean_values(i, df):
    title = df.columns[i]
    x = df.iloc[:,0]
    y = df.iloc[:,i].copy(deep = True)
    y1 = (y - np.mean(y))/np.std(y)
    ma_1 = y1.rolling(window = 5*24*60).mean()
    plt.plot(x, y1, '--b')
    plt.plot(x, ma_1, '--r')
    plt.title(title)
    plt.xticks(rotation = 'vertical')
    plt.show()
    return
        
def plot_all_mv(df):
    x = df.iloc[:,0]    
    for i in [3,10, 8]:
        label = df.columns[i]     
        y = df.iloc[:,i].copy(deep = True)
        y1 = (y - np.mean(y))/np.std(y)
        ma_1 = y1.rolling(window = 5*24*60).mean()
        plt.plot(x, ma_1, label = label)
        plt.axhline(y=0, color = 'k')
    plt.xticks(rotation = 'vertical')
    plt.legend()
    plt.show()
    return

def plot_histogram_with_alarm_limits(i, df):
    y = df.iloc[ind,i].values.copy()
    y = np.array(y, dtype='float')
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
    plt.ylim(mean - 20*sd, mean + 20*sd)
    plt.title(df.columns[i])  
    plt.legend()
    plt.show()
    return
    

ind = get_index_in_single_array(indices_to_keep)
plotting_clean_values(2, cleaned_df, ind)
plotting_mv_clean_values(4, cleaned_df)
plot_all_mv(cleaned_df)
mean, sd = plot_histogram_with_alarm_limits(i = 4, df = cleaned_df)
plot_alarm_limit_on_ts(4, cleaned_df, mean, sd)
    
###############################################################################
# plot histogram showing alarm limits
###############################################################################




    
    
    