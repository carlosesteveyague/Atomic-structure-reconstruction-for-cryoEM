#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:06:13 2022

@author: ce423
"""

import torch

from data_generator.structure_batch import chain_structure

import torch.nn as nn



def dataset(dataset_features):
    
    struct = dataset_features['struct']
    n_imgs = dataset_features['n_imgs']
    orient_variability = dataset_features['orient_variability']
    
    n_px_3d = dataset_features['n_px_3d']
    sigma_gaussian_3d = dataset_features['sigma_gaussian_3d']
    mask_size_3d = dataset_features['mask_size_3d']
    noise_3d = dataset_features['noise_3d']
    
    CTF_data_feat = dataset_features['CTF_data_feat']
    
    # We compute the curves, and the volumes in the trajectory
    
    X = struct.discrete_curves()
    
    X_scaled, mask_3d = pointcloud_scaling_mask(X, n_px_3d, sigma_gaussian_3d, mask_size_3d)
    X_scaled = X_scaled.round().long()
    
    
    convolution = nn.Conv3d(1, 1, mask_size_3d, bias = False, padding = 'same')
    convolution.weight = nn.Parameter(mask_3d)
    
    volumes = torch.zeros([X.shape[0], 1, n_px_3d, n_px_3d, n_px_3d])
    
    for j in range(X_scaled.shape[0]):
        for i in range(X_scaled.shape[1]):
            volumes[j,0, X_scaled[j,i,0], X_scaled[j,i,1], X_scaled[j,i,2]] = 1.
    
    with torch.no_grad():
        volumes = convolution(volumes).squeeze(1)
    
    ## We now take a random sampling from the trajectory
    
    idx_sample = torch.randint(struct.Theta.shape[0], [n_imgs])
    
    vols_data = volumes[idx_sample]
    
    vols_data = vols_data + noise_3d*torch.randn(vols_data.shape)
    
    
    
    Psi_sample = struct.Psi[idx_sample]
    Theta_sample = struct.Theta[idx_sample]
    x0_sample = struct.x0[idx_sample]
    orientations_sample = struct.orientation[idx_sample]
    dists_sample = struct.dists[idx_sample]
    ref_idx = struct.ref_idx
    
    struct_data = chain_structure(Psi_sample, Theta_sample, x0_sample, orientations_sample, dists_sample, ref_idx)
      
    orient0_diff = orient_variability[0,0] + (orient_variability[0,1] - orient_variability[0,0])*torch.rand([n_imgs, 1])
    orient1_diff = orient_variability[1,0] + (orient_variability[1,1] - orient_variability[1,0])*torch.rand([n_imgs, 1])
    orient2_diff = orient_variability[2,0] + (orient_variability[2,1] - orient_variability[2,0])*torch.rand([n_imgs, 1])
    
    orientation_diffs_data = torch.cat([orient0_diff, orient1_diff, orient2_diff], dim = -1)
    
    
    n_px_2d = dataset_features['n_px_2d']
    noise_2d = dataset_features['noise_2d']
    density_type = dataset_features['density_type']
    sigma_density = dataset_features['sigma_density']
    
    CT_images_noise, CT_images_clean, clean_imgs_no_CTF, img_lims, Df = struct_data.CT_images(orientation_diffs_data, n_px_2d, noise_2d, density_type, sigma_density, CTF_data_feat)
    
    return vols_data, struct_data, orientation_diffs_data, CT_images_noise, CT_images_clean, clean_imgs_no_CTF, img_lims, Df


def pointcloud_scaling_mask(X, n_px_3d, sigma_gaussian_3d, mask_size_3d):
    
    Xmax = X.max()
    Xmin = X.min()
    
    scaling = .7*n_px_3d/(Xmax-Xmin)
    
    grid = torch.arange(-(mask_size_3d-1)/2, (mask_size_3d-1)/2 + 1)/scaling
    
    gridX, gridY, gridZ = torch.meshgrid([grid,grid,grid])
    
    mask = torch.exp(-(gridX**2 + gridY**2 + gridX**2)/(2*sigma_gaussian_3d**2))
    
    return n_px_3d*(.7*(X-Xmin)/(Xmax-Xmin) + .15), mask.unsqueeze(0).unsqueeze(0)