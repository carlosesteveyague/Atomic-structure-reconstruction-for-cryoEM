#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 19:28:56 2022

@author: carlosesteveyague
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

def disp_point_cloud(coords):
    
    assert (coords.shape[1]==2) or (coords.shape[1] == 3)
    
    coords = np.array(coords)
    
    ## coords must be an array with dims [n, 2] or [n, 3]
    ## where n is the number of points
    
    if coords.shape[1] == 2:
        im = plt.scatter(coords[:,0], coords[:,1])
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        plt.show()
    
    if coords.shape[1] == 3:
        
        ax = plt.axes(projection='3d')
        xdata = coords[:,0]
        ydata = coords[:,1]
        zdata = coords[:,2]
        ax.scatter(xdata, ydata, zdata,c = zdata, cmap = 'blues')
        ax.set_axis_off() 
        plt.show()
