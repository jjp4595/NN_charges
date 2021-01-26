# -*- coding: utf-8 -*-
"""
This takes the spherical data and gets it into CSV format
"""

import os
import numpy as np
from scipy.signal import savgol_filter
import glob
import pandas as pd


def standoff_func(fileIN): #This calculates scaled distance by subtracting the charge center from y domain max, so this may have to change. 
    with open(fileIN) as f:
        content = f.readlines()
        content = [x.strip() for x in content] #remove whitespace at end of each line. 
    #Getting charge mass
    charge_info = np.core.defchararray.find(content, "use charge, main")
    index_charge = np.argwhere(charge_info > 0)
    charge_mass=[]
    for word in content[index_charge[0,0]].split():
        try:
            charge_mass.append(float(word))
        except ValueError:
            pass
   #Getting Standoff info
    chargecenter_info = np.core.defchararray.find(content, "charge center")
    index_chargecenter = np.argwhere(chargecenter_info > 0)
    chargecenter=[]
    for word in content[index_chargecenter[0,0]].split():
        try:
            chargecenter.append(float(word))
        except ValueError:
            pass
   #Getting edge boundary info
    bound_index = 24 #Change this if row that model info is on changes!
    bound_info=[]
    for word in content[bound_index].split():
        try:
            bound_info.append(float(word))
        except ValueError:
            pass
    bound = bound_info[4] #This time we are interested in y domain. 
    standoff = bound - chargecenter[1] 
    return standoff

def FileAddressList(fileIN, GaugeData=None):
    #This functions reads the fileIN and then creates the filelist as a list of strings from that recursive search
    #If a second argument is entered, then it will import the gauge data. 
    #In other words, one input for file lists, add a number for gauge data import.
    OutputList=[]
    if GaugeData is None:
        for filename in glob.glob(fileIN, recursive=True):
            OutputList.append(filename)
    else: 
        for filename in glob.glob(fileIN, recursive=True):
            OutputList.append(np.loadtxt(filename))
    return OutputList

charge_rad = 0.0246
cm = 0.1
TNTeq = 1


def dataimport(folderpath, cm, TNTeq, sav=151):
    file = FileAddressList(folderpath + r"\*.txt")
    data = FileAddressList(folderpath+ r"\*gtable", 1)
    data = np.asarray([data[i][:,7] for i in range(len(file))]).T 
    smooth = np.asarray([savgol_filter(data[:,i], sav, 3) for i in range(len(file))]).T
    smooth_Icr = np.asarray([smooth[:,i]/ (max(smooth[:,i])) for i in range(len(file))]).T     
    z_center = [(standoff_func(file[i]))/((cm*TNTeq)**(1/3)) for i in range(len(file))]
    z_clear = [(standoff_func(file[i]) - charge_rad)/((cm*TNTeq)**(1/3)) for i in range(len(file))]
    z_center = np.asarray(z_center)
    z_clear = np.asarray(z_clear)
    so = (np.asarray(z_center) * ((cm*TNTeq)**(1/3)))    
    so_clear = (np.asarray(z_clear) *((cm*TNTeq)**(1/3)))  
    keys = ['imp', 'imp_smooth', 'icr', 'z', 'z_clear', 'so', 'so_clear']
    vals = [data, smooth, smooth_Icr, z_center, z_clear, so, so_clear]
    d = dict(zip(keys, vals))
    return d
large = dataimport(os.environ['USERPROFILE'] + r"\Google Drive\Apollo Sims\Impulse Distribution Curve Modelling\Paper_1\Sphere\main_z16_5", cm, TNTeq)




#create pandas dataframe from dict object
cols = ['mass', 'so', 'theta', 'imp']

mass = np.repeat(0.1, 3600)
so = np.repeat(large['so'], 200)
z = np.repeat(large['z'], 200)
imp = large['imp_smooth'].flatten('F')
theta = np.tile(np.linspace(0,80,200), 18)


new = np.stack((mass, so, z, theta, imp), axis = 1)
np.savetxt('spherical.csv', new, delimiter=',', fmt='%.5f')





#create some validation data
val_highZ = dataimport(os.environ['USERPROFILE'] + r"\Google Drive\Apollo Sims\Impulse Distribution Curve Modelling\Paper_1\Sphere\validation_samples\250kg\res5", 250,TNTeq, sav=151)
mass = np.repeat(250, 200)
so = np.repeat(val_highZ['so'], 200)
z = np.repeat(val_highZ['z'], 200)
imp = val_highZ['imp_smooth'].flatten('F')
theta = np.linspace(0,80,200)
new = np.stack((mass, so, z, theta, imp), axis = 1)
np.savetxt('spherical_val_highZ.csv', new, delimiter=',', fmt='%.5f')

val_lowZ = dataimport(os.environ['USERPROFILE'] + r"\Google Drive\Apollo Sims\Impulse Distribution Curve Modelling\Paper_1\Sphere\validation_samples\5kg\res4", 5,TNTeq, sav=151)
mass = np.repeat(5, 200)
so = np.repeat(val_lowZ['so'], 200)
z = np.repeat(val_lowZ['z'], 200)
imp = val_lowZ['imp_smooth'].flatten('F')
theta = np.linspace(0,80,200)
new = np.stack((mass, so, z, theta, imp), axis = 1)
np.savetxt('spherical_val_lowZ.csv', new, delimiter=',', fmt='%.5f')

