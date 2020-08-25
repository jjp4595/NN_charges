# -*- coding: utf-8 -*-
"""
This takes the spherical data and gets it into CSV format
"""

import os
import numpy as np
from scipy.signal import savgol_filter
import glob
import pandas as pd
import scipy.integrate as si
import re


def so_mass_l_d(fileIN): 
    with open(fileIN) as f:
        content = f.readlines()
        content = [x.strip() for x in content]  
        

        #Domain
        str_to_search = "\t\t\t\t...zone length, origin"
        str_index = np.argwhere(np.core.defchararray.find(content, str_to_search) > 0)
        boundary = re.split(' ', content[int(str_index + 2)]) 
        boundary = float(boundary[7])

        #Charge  
        str_to_search = "\t\t\t...charge center, axis vector"
        term_time_index = np.argwhere(np.core.defchararray.find(content, str_to_search) > 0)
        loc = re.split(' ', content[int(term_time_index)])
        loc = float(loc[1])
        so = round(boundary - loc,3)
        
        mass = np.core.defchararray.find(content, "...charge mass, shape, L/D")
        index = np.argwhere(mass > 0) 
        mass, l_d = float(content[index[0,0]].split()[0]), float(content[index[0,0]].split()[2])    
        
        
    return so, mass, l_d

def fromapollo(fileIN, GaugeData=None):
    #This functions reads the fileIN and then creates the filelist as a list of strings from that recursive search
    #If a second argument is entered, then it will import the gauge data. 
    #In other words, one input for file lists, add a number for gauge data import.
    OutputList=[]
    if GaugeData is None:
        for filename in glob.glob(fileIN, recursive=True):
            OutputList.append(filename)
    else: 
        for filename in glob.glob(fileIN, recursive=True):
            OutputList.append(np.genfromtxt(filename))#changed to genfromtxt as txt file has strings in it
    return OutputList


def dataimport(folderpath, cm, TNTeq, sav=41):
    file = fromapollo(folderpath + r"\*.txt")
    data = fromapollo(folderpath+ r"\*gtable", 1)
    
    data = np.asarray([data[i][:,-1] for i in range(len(file))]).T
    smooth_gtable = np.asarray([savgol_filter(data[:,i], sav, 3) for i in range(len(file))]).T
    
    gauges = fromapollo(folderpath+ r"\*gauges", 1)    
    max_spec_imps=[]
    for i in range(len(file)):
        max_spec_imps.append([si.cumtrapz(gauges[i][:,j]-101300, gauges[i][:,0], initial=0).max() for j in range(1,201)]) 
    maximps = np.asarray(max_spec_imps).T
    
    smooth = np.asarray([savgol_filter(maximps[:,i], sav, 3) for i in range(len(file))]).T
    smooth_Icr = np.asarray([smooth[:,i]/ (max(smooth[:,i])) for i in range(len(file))]).T     
    
    so, mass, l_d = [],[],[]
    for f in file:
        a, b, c = so_mass_l_d(f)
        so.append(a) 
        mass.append(b)
        l_d.append(c)
    so = np.asarray(so)
    mass = np.asarray(mass)
    l_d = np.asarray(l_d)
    
    so, mass, l_d = so.reshape(1,len(file)), mass.reshape(1,len(file)), l_d.reshape(1,len(file))
    so, mass, l_d = np.repeat(so, len(smooth), axis=0), np.repeat(mass, len(smooth), axis=0), np.repeat(l_d, len(smooth), axis=0)
    
    keys = ['imp_gtable', 'imp_smooth_gtable', 'gauges', 'imp_smooth', 'icr', 'peak_i_from_gauges', 'so', 'mass', 'l_d']
    vals = [data, smooth_gtable, gauges, smooth, smooth_Icr, maximps, so, mass, l_d]
    d = dict(zip(keys, vals))
    return d



#import data
folderz = ['\_0_1', '\_0_25', '\_0_5', '\_0_75', '\_1', '\_2', '\_3']
z = np.round(np.linspace(0.1,0.5,5)[1::],3)
rootPath = os.environ['USERPROFILE'] + r"\Google Drive\Apollo Sims\Impulse Distribution Curve Modelling\Paper_2\Dataset_ZL100mm_res2\Dataset_ZL100mm_res2"
cylinders  = {a:dataimport(rootPath+a, 1, 1, sav=51) for a in folderz}



#create pandas dataframe from dict object
features = ['mass', 'so', 'theta', 'l_d', 'imp']

def datapull(keyIN):
    shapes = np.shape(cylinders[keyIN]['imp_smooth_gtable'])
    
    mass = cylinders[keyIN]['mass'].flatten('F')
    so = cylinders[keyIN]['so'].flatten('F')
    theta = np.tile(np.linspace(0,80,200), shapes[1])
    l_d = cylinders[keyIN]['l_d'].flatten('F')
    imp = cylinders[keyIN]['imp_smooth_gtable'].flatten('F')
    new = np.stack((mass, so, theta, l_d, imp), axis = 1)
    return new

data = [datapull(k) for k in cylinders]
data = np.concatenate(data, axis = 0)

np.savetxt('cylindrical.csv', data, delimiter=',', fmt='%.5f')



