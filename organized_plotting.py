# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:34:22 2018

@author: Ravi Tiwari
"""

def plot_ts(i, df, col = 'b', ind = None, ax = None, dates = None):
    if ax is None:
        f, ax = plt.subplots()
    x = df.iloc[:,0]
    y = df.iloc[:,i].copy(deep = True)
    if ind is not None:
        inv_ind = np.invert(ind)
        y[inv_ind] = None
    ax.plot(x, y, color = col) 
    plt.xticks(rotation = 'vertical')
    
    return ax


def plot_ts_date_subset(i, dates, margin, ind, col, df, ax):
    ind1 = df.iloc[:,0]  > start_date
    ind2 = df.iloc[:,0]  < end_date
    date_ind = ind1 & ind2
    x = df.loc[date_ind].iloc[:,0]
    y = df.loc[date_ind].iloc[:,i].copy(deep = True)
    inv_ind = np.invert(ind)
    y[inv_ind] = None
    ax.plot(x,y, color = col)
    return ax

def set_x_limit(start_date, end_date, ax):
    ax.set_xlim(start_date, end_date)
    return ax   

def set_y_limit(low, high, ax):
    ax.set_ylim(low, high)
    return ax

def add_y_line(y, ax):
    for value in y:
        ax.axhline(y = value, color = 'k', linewidth = 1, linestyle = '--')
    return ax

def add_y_label(label, ax, col = 'k'):
    ax.set_ylabel(label, color = col)
    ax.tick_params('y', colors=col)
    return ax

def add_x_label(label, ax):
    ax.set_xlabel(label)
    return ax

def create_twin_axis(ax):
    ax0 = ax.twinx()
    return ax0

def add_change_points(change_point, df, ax):    
    ind = change_point.index
    x_change = df.loc[ind].iloc[:,0]
    for t, ct in zip(x_change, change_point):
        if ct == 1:
            col = 'red'
        if ct == -1:
            col = 'green'
        ax.axvline(x = t, color = col, linewidth = 1, linestyle = '--')    
    return ax
    

def plot_histogram_with_alarm_limits(y, mean, sd, ax = None):
    if ax is None:
        f, ax = plt.subplots()
    hh = round(mean + 3*sd, 2)
    ll = round(mean - 3*sd, 2)
    sns.distplot(y, bins = 30, color = 'green', vertical = True, ax = ax)
    ax.axhline(y = mean, color = 'k', linestyle = '--')
    ax.axhline(y = hh, color = 'red', linestyle = '--')
    ax.axhline(y = ll, color = 'red', linestyle = '--')
    ax.text(0.1, 0.9, 'HH: ' + str(hh), transform=ax.transAxes)
    ax.text(0.1, 0.8, 'LL: ' + str(ll), transform=ax.transAxes)
    return ax

def plot_alarm_limit_on_ts(x, y, mean, sd, ax = None):
    if ax is None:
        f, ax = plt.subplots()
        
    lower = mean - 3*sd
    upper = mean + 3*sd
    
    youtside = np.ma.masked_inside(y, lower, upper)
    yinside = np.ma.masked_outside(y, lower, upper)
    
    ax.plot(x, youtside, 'red', label = 'Abnormal')
    ax.plot(x, yinside, 'green', label = 'Normal')
         
    ax.axhline(y=lower, color = 'red', linestyle='--')
    ax.axhline(y=mean, color = 'k', linestyle='--')
    ax.axhline(y=upper, color = 'red', linestyle='--')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim(mean - 5*sd, mean + 5*sd)
#    ax.legend()
    return ax


def plot_complete_time_series(i, df, col, indices = None, ax = None):
    for j in range(3):
        sdf = df.loc[j]
        if indices is not None:
            ind = indices[j]
            if ax is None:
                ax = plot_ts(i, sdf, col = col, ind = ind)
            else:
                ax = plot_ts(i, sdf, col = col, ind = ind, ax = ax)
        else:
            if ax is None:
                ax = plot_ts(i, sdf, col = col)
            else:
                ax = plot_ts(i, sdf, col = col, ax = ax)       
    return ax

def plot_complete_time_series_subset(i, dates, margin, indices, col, df, ax):
    for j in range(3):
        sdf = df.loc[j]
        ind = indices[j]
        ax = plot_ts_date_subset(i, dates, margin, ind, col, sdf, ax)  
    return ax
    

###############################################################################
# Step by step plotting
###############################################################################
# Step 1: The whole plot
i = 1
ylabel = cleaned_abnormal_df.columns[i]
start_date = '2018-04-12 00:00:00'
end_date = '2018-12-07 00:00:00'

load = [90, 100]
start_date = '2014-04-12 00:00:00'
end_date = '2015-12-07 00:00:00'
dates = [start_date, end_date]
margin = pd.Timedelta('5 days')
abnormal_input_data, indices = get_input_data_for_alarm_setting(i, load, margin, cleaned_abnormal_df, dates)
mean, sd = get_mean_sd(abnormal_input_data)

i = 1
col = 'blue'
ax = plot_complete_time_series(i, cleaned_abnormal_df, col)
ylabel = cleaned_abnormal_df.columns[i]
ax = add_y_label(ylabel, ax, col)
ax = add_y_line(load, ax)
i = 4
col = 'red'
ylabel = cleaned_abnormal_df.columns[i]
ax0 = create_twin_axis(ax)
ax0 = plot_complete_time_series(i, cleaned_abnormal_df, col, indices, ax0)
ax0 = add_y_label(ylabel, ax0, col)

ax0 = plot_complete_time_series_subset(i, dates, margin, indices, 'green', cleaned_abnormal_df, ax0)
#ax = set_x_limit(start_date, end_date, ax0)
###############################################################################
# Step 2: add the data that is used to determine the alarm limit
# Added in the previous step


###############################################################################
# Step 3a: Show alarm limit calculations
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
for item in abnormal_input_data:
    ax = plot_alarm_limit_on_ts(item[0], item[1], mean, sd, ax1)
ax1 = set_x_limit(start_date, end_date, ax)

# Step 3b: histogram generation
y_clean = []
for item in abnormal_input_data:
    y_clean.extend(item[2])
y_clean = np.array(y_clean)

ax2 = plot_histogram_with_alarm_limits(y_clean, mean, sd, ax2)

###############################################################################
# Step 4: show alarm results on all the data
###############################################################################












###############################################################################
# Step 1: plot the subsetted data
###############################################################################
i = 4
print(cleaned_abnormal_df.columns[i])
load = [90, 100]
start_date = '2014-04-12 00:00:00'
end_date = '2018-12-07 00:00:00'
dates = [start_date, end_date]
margin = pd.Timedelta('5 days')
abnormal_input_data, indices = get_input_data_for_alarm_setting(i, load, margin, cleaned_abnormal_df, dates)
mean, sd = get_mean_sd(abnormal_input_data)


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
plot_alarm_limit_on_ts(x, y, mean, sd, ax1)
plot_histogram_with_alarm_limits(y_clean, mean, sd, ax2)





i = 5
print(cleaned_abnormal_df.columns[i])
load = [90, 100]
start_date = '2014-04-12 00:00:00'
end_date = '2015-06-07 00:00:00'
dates = [start_date, end_date]
margin = pd.Timedelta('5 days')
abnormal_input_data, indices = get_input_data_for_alarm_setting(i, load, margin, cleaned_abnormal_df, dates)
ind = indices[0]
change_point = get_change_point_location(cleaned_abnormal_df.loc[0], load)
ax = plot_ts(1, cleaned_abnormal_df.loc[0], ax = None)
ax = add_y_line(load[0], ax)
ax = add_y_line(load[1], ax)
ax = add_change_points(change_point, cleaned_abnormal_df.loc[0], ax)
#ax = set_x_limit(start_date, end_date, ax)
ax = plot_values_in_specified_load_and_margin(i, ind, cleaned_abnormal_df.loc[0], ax)
plot_histogram_with_alarm_limits(y, mean, sd, ax = None)
ax = set_x_limit(start_date, end_date, ax)
#

###############################################################################
len(abnormal_input_data)
for item in abnormal_input_data:
    print(len(item[0]), len(item[1]), len(item[2]))








