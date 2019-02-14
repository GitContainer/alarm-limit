# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:03:38 2019

@author: Ravi Tiwari
"""

import os

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
# 









