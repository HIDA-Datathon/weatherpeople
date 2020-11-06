#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:05:21 2020

@author: felix
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(output_path = '../Data/nao_index_train.npy',
              x1_path = '../Data/tas_train.npy',
              x2_path = '../Data/psl_train.npy'):
    '''
    Parameters
    ----------
    output_path : TYPE, optional
        DESCRIPTION. The default is './Data/nao_index_train.npy'.
    x1_path : TYPE, optional
        DESCRIPTION. The default is './Data/tas_train.npy'.
    x2_path : TYPE, optional
        DESCRIPTION. The default is './Data/psl_train.npy'.

    Returns
    -------

    x1 : North Atlantic and Tropical Atlantic near surface air temperature, 
        October-November average (one vector of dimension M1 per year= instance)
    x2 : North Atlantic sea-level-pressure, October-November average 
        (one vector of dimension M2 per instance), Shape(N,M2)
    y : North Atlantic Oscillation Index

    '''
    
    y = np.load(output_path)
    x1 = np.load(x1_path)
    x2 = np.load(x2_path)
    return x1, x2, y




x1, x2, y = load_data()
x1 = x1.reshape((900,54,43))
x2 = x2.reshape((900,54,43))


corr_temp = np.zeros((54,43))
corr_press = np.zeros((54,43))

for i in range(54):
    for j in range(43):
        corr_temp[i,j] = np.corrcoef(x1[:,i,j],y)[0,1]
        corr_press[i,j] = np.corrcoef(x2[:,i,j],y)[0,1]



x = np.arange(-78.75, -78.75+ 1.875*43, 1.875 )
y = np.arange(-19.5852186088223,-19.5852186088223+54*1.865, 1.865)

X, Y = np.meshgrid(x, y)


fig,ax=plt.subplots()
plt.title("Correlation Temperature - NAO")
cs1 = ax.contourf(X,Y,corr_temp,vmin=-0.13,vmax=0.13, cmap='RdBu',levels=np.arange(-0.15,0.18,step=0.03))
plt.ylabel("Latitude")
plt.xlabel("Longitude")
fig.colorbar(cs1)
plt.show()

fig,ax=plt.subplots()
plt.title("Correlation Pressure - NAO")
cs = ax.contourf(X,Y,corr_press,vmin=-0.13,vmax=0.13, cmap='RdBu',levels=np.arange(-0.15,0.18,step=0.03))
plt.ylabel("Latitude")
plt.xlabel("Longitude")
fig.colorbar(cs)
plt.show()
