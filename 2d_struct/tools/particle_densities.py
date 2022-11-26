#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:35:41 2022

@author: carlosesteveyague
"""

import numpy as np
import torch

def density_comp(X, Y, Gamma, sigma, density_type = 'Gaussian'):
    
    Gamma_ext = Gamma.unsqueeze(-1).unsqueeze(-1)
    
    X_diff = X - Gamma_ext[:,0]
    Y_diff = Y - Gamma_ext[:,1]
    
    if density_type == 'Gaussian':
       exps = torch.exp(-(X_diff**2 + Y_diff**2)/(2*sigma**2))
       return torch.sum(exps, dim = 0)
    
    elif density_type == 'Charac function':
        dists_sq = X_diff**2 + Y_diff**2
        out = torch.zeros(dists_sq.shape, dtype = torch.float)
        out[dists_sq <= sigma**2] = 1.
        return out.sum(dim = 0)
    
    elif density_type == 'Bang-bang':
        return 1.*((X_diff**2 + Y_diff**2).min(dim = 0)[0] <= sigma**2)
    
    if density_type == 'Gaussian max normalized':
       exps = torch.exp(-(X_diff**2 + Y_diff**2)/(2*sigma**2))
       return torch.max(exps, dim = 0)[0]
    
    

def density_comp_batch(X, Y, Gamma, sigma, density_type = 'Gaussian'):
    
    XY = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], -1)
    
    XY_ext = XY.unsqueeze(0).repeat(Gamma.shape[0], 1, 1, 1)
    
    v_size = XY_ext.shape[1]*XY_ext.shape[2]
    grid_list = XY_ext.reshape(Gamma.shape[0], v_size, 2)
    
    dists = torch.cdist(Gamma, grid_list)
    
    if density_type == 'Gaussian':
        exps = torch.exp(-dists**2/(2*sigma**2))
        return exps.sum(dim=1).reshape(Gamma.shape[0], XY.shape[0], XY.shape[1])
    
    elif density_type == 'Charac function':
        out = torch.zeros(dists.shape, dtype = torch.float)
        out[dists <= sigma] = 1.
        return out.sum(dim = 1).reshape(Gamma.shape[0], XY.shape[0], XY.shape[1])
    
    elif density_type == 'Bang-bang':
        dists = dists.min(dim = 1)[0]
        out = torch.zeros(dists.shape)
        out[dists <= sigma] = 1. 
        return out.reshape(Gamma.shape[0], XY.shape[0], XY.shape[1])
    
    if density_type == 'Gaussian max normalized':
        exps = torch.exp(-dists**2/(2*sigma**2))
        return exps.max(dim=1)[0].reshape(Gamma.shape[0], XY.shape[0], XY.shape[1])
    


def RadonTransform_Gaussians(X, Gamma, sigma):
    
    X_ext = X.unsqueeze(0).repeat(Gamma.shape[0],1).unsqueeze(-1)
    
    Gamma1 = Gamma[:,:,0].unsqueeze(-1)
    
    dists = torch.cdist(Gamma1, X_ext)
    
    #exps = sigma*np.sqrt(2*np.pi)*torch.exp(-dists**2/(2*sigma**2))
    exps = torch.exp(-dists**2/(2*sigma**2))
    
    return exps.sum(dim=1)
    







