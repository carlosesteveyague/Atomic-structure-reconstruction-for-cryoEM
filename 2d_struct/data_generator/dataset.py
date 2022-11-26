#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:26:09 2022

@author: carlosesteveyague
"""

import torch


def dataset_gen(dataset_features, loop = True):
    
    struct = dataset_features['struct']
    
    n = dataset_features['n_imgs']
    
    scaling = 2*torch.pi*torch.Tensor(dataset_features['flex_var']).unsqueeze(0)
    
    diffs_data = torch.rand([n, 3], dtype = torch.float)*scaling - scaling/2
    
    Theta_diffs = torch.zeros([n, struct.Theta.shape[0]])
    
    Theta_diffs[:, struct.hand1_theta_idx] = diffs_data[:,0]
    Theta_diffs[:, struct.hand2_theta_idx] = diffs_data[:,1]
    
    Theta_diffs[:, struct.box_corner_idxs[0]] = diffs_data[:,2]/2
    Theta_diffs[:, struct.box_corner_idxs[1]] = -diffs_data[:,2]/2
    Theta_diffs[:, struct.box_corner_idxs[2]] = diffs_data[:,2]/2
    Theta_diffs[:, struct.box_corner_idxs[3]] = -diffs_data[:,2]/2
    
    #noise_theta = 0.01*torch.randn(struct.Theta.shape)
    #struct.Theta += noise_theta 
    
    ## Generate 2d images
    noise = dataset_features['noise']
    density_type = dataset_features['density_type']
    sigma_density = dataset_features['sigma_density']
    n_px = dataset_features['n_px']     
    
    if loop == True:
        imgs_2d = struct.get_images_loop(n_px, Theta_diffs, noise, density_type, sigma_density)
    else:
        imgs_2d = struct.get_images(n_px, Theta_diffs, noise, density_type, sigma_density)
    
    ## Generate 1d CT projections
    
    noise_CT = dataset_features['noise_CT']
    sigma_density_CT = dataset_features['sigma_density_CT']
    n_px_CT = dataset_features['n_px_CT']
    
    alpha_orient = 2*torch.pi*torch.rand(n)
    
    orientations = torch.zeros([n,2])
    
    orientations[:,0] = torch.cos(alpha_orient)
    orientations[:,1] = torch.sin(alpha_orient)
    
    CT_projs = struct.get_CT_images(n_px_CT, Theta_diffs, noise_CT, sigma_density_CT, orientations)
    
    return imgs_2d, CT_projs, Theta_diffs, orientations
    
    
    