#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:04:23 2022

@author: ce423
"""
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

def disp_image(img, multi_img = False, colormap = 'Greys'):
    img = np.array(img)
    
    if len(img.shape) > 2:
        img = img.reshape(-1, img.shape[-2], img.shape[-1])
        
        if multi_img == True:
            fig, axs = plt.subplots(1,img.shape[0])
            for i in range(img.shape[0]):
                axs[i].imshow(img[i], cmap = colormap)
                axs[i].axes.get_xaxis().set_visible(False)
                axs[i].axes.get_yaxis().set_visible(False)
            plt.show()
            
        else:
            for i in range(img.shape[0]):
                im = plt.imshow(img[i], cmap = colormap)
                im.axes.get_xaxis().set_visible(False)
                im.axes.get_yaxis().set_visible(False)
                plt.show()
    else:
        im = plt.imshow(img, cmap = 'Greys')
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        plt.show()

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


def disp_img_validation(init_imgs, ground_truth_imgs, pred_imgs):
    
    for i in range(pred_imgs.shape[0]):
        fig, axs = plt.subplots(1, 3)
        
        axs[0].imshow(init_imgs[i].transpose(0,1).flip(0,), cmap='Greys')
        axs[0].set_title('Initialization')
        
        
        axs[1].imshow(ground_truth_imgs[i].transpose(0,1).flip(0,), cmap='Greys')
        axs[1].set_title('Ground truth')
        
        
        axs[2].imshow(pred_imgs[i].transpose(0,1).flip(0,), cmap='Greys')
        axs[2].set_title('Prediction')
        
        name = 'Figures_paper/Fig_3d_predictions/image_pred%i' %i
        #plt.savefig(name)
        plt.show()

def disp_struct_validation(init_struct, ground_truth_struct, pred_struct):

    for i in range(pred_struct.shape[0]):
        
        fig = plt.figure(figsize=plt.figaspect(0.3))
        
        
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        xdata = np.array(init_struct[i,:,0])
        ydata = np.array(init_struct[i,:,1])
        zdata = np.array(init_struct[i,:,2])
        ax.plot3D(xdata, ydata, zdata, 'black')
        ax.scatter(xdata, ydata, zdata,c = zdata, cmap = 'blues')
        ax.set_axis_off() 
        ax.set_title('Initialization')
        
        
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        xdata = np.array(ground_truth_struct[i,:,0])
        ydata = np.array(ground_truth_struct[i,:,1])
        zdata = np.array(ground_truth_struct[i,:,2])
        ax.plot3D(xdata, ydata, zdata, 'black')
        ax.scatter(xdata, ydata, zdata,c = zdata, cmap = 'blues')
        ax.set_axis_off() 
        ax.set_title('Original')
        
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        xdata = np.array(pred_struct[i,:,0])
        ydata = np.array(pred_struct[i,:,1])
        zdata = np.array(pred_struct[i,:,2])
        ax.plot3D(xdata, ydata, zdata, 'black')
        ax.scatter(xdata, ydata, zdata,c = zdata, cmap = 'blues')
        ax.set_axis_off() 
        ax.set_title('Prediction')
        
        name = 'Figures_paper/Fig_3d_predictions/atomic_model_pred%i' %i
        #plt.savefig(name)
        plt.show()
