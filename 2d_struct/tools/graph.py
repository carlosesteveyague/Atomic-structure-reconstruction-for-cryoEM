#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:52:52 2022

@author: carlosesteveyague
"""

import numpy as np
import torch

def graph_gaussian_kernel(data, sigma_gauss_kernel):
    v_size = data.shape[1]*data.shape[2]
    data = data.reshape([data.shape[0], v_size])
    
    dists = torch.cdist(data,data, compute_mode= 'donot_use_mm_for_euclid_dist')
    
    W = torch.exp(-dists**2/(2*sigma_gauss_kernel**2))
    
    for i in range(W.shape[0]):
        W[i,i] = 0
    
    return W



def laplacian(W, normalize = True):
    
    Dg = W.sum(dim = 0).diag()
    
    L = Dg - W
    
    if normalize == True:
        Dg_inv = Dg.clone()
        Dg_inv[Dg>0] = Dg[Dg>0]**(-0.5)
        
        L = torch.mm(Dg_inv, torch.mm(L, Dg_inv))
    
    return L

def eigen_graph(Lap, n_eig):
    
    Lap = np.array(Lap)
    [eigen_val, eigen_vec] = np.linalg.eig(Lap)
    
    idx_sorted = np.argsort(eigen_val)[:n_eig]
    
    eigen_val = eigen_val[idx_sorted]
    eigen_vec = eigen_vec[:,idx_sorted]
    
    if eigen_vec[0,0] < 0:
        eigen_vec[:, 0] = -eigen_vec[:, 0]
    
    return torch.Tensor(eigen_val), torch.Tensor(eigen_vec)

    

