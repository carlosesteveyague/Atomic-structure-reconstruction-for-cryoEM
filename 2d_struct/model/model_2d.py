#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:58:20 2022

@author: carlosesteveyague
"""

import torch
import torch.nn as nn



class model_2dCT(nn.Module):
    
    def __init__(self, model_features):
        super(model_2dCT, self).__init__()
        
        self.struct = model_features['struct']
        self.param_idxs = model_features['param_idxs']
        n_eigenval = model_features['n_eigenval']
        
        self.Gamma = nn.Linear(n_eigenval, len(self.param_idxs), bias = False)
        
        self.Gamma.weight = nn.Parameter(torch.zeros([len(self.param_idxs), n_eigenval]))
        
        self.density_type = model_features['density_type']
        self.sigma_density = model_features['sigma_density']
        self.n_px = model_features['n_px']
        
        self.sigma_density_CT = model_features['sigma_density_CT']
        self.n_px_CT = model_features['n_px_CT']
        
        
    def forward(self, inputs):
        
        V = inputs[:,:-2]
        orientations = inputs[:,-2:]
        
        Theta_diffs = torch.zeros(V.shape[0], self.struct.Theta.shape[0])
        
        Theta_diffs[:, self.param_idxs] = self.Gamma(V)
        
        return self.struct.get_CT_images(self.n_px_CT, Theta_diffs, 0, self.sigma_density_CT, orientations)
    
    def forward_BB(self, inputs):
        V = inputs[:,:-2]
        
        Theta_diffs = torch.zeros(V.shape[0], self.struct.Theta.shape[0])
        
        Theta_diffs[:, self.param_idxs] = self.Gamma(V)
        
        return self.struct.compute_BB(Theta_diffs)
    
    def forward_2d(self, inputs):
        V = inputs[:,:-2]
        orientations = inputs[:,-2:]
        
        Theta_diffs = torch.zeros(V.shape[0], self.struct.Theta.shape[0])
        
        Theta_diffs[:, self.param_idxs] = self.Gamma(V)
        
        return self.compute_images(Theta_diffs, True, orientations)
    
    def get_diffs(self, V):
        
        Theta_diffs = torch.zeros(V.shape[0], self.struct.Theta.shape[0])
        Theta_diffs[:, self.param_idxs] = self.Gamma(V)
        return Theta_diffs
        
    def compute_images(self, Theta_diffs, loop, orientations = None):
        
        if loop == True:
            return self.struct.get_images_loop(self.n_px, Theta_diffs, 0, self.density_type, self.sigma_density, orientations)
        else:
            return self.struct.get_images(self.n_px, Theta_diffs, 0, self.density_type, self.sigma_density, orientations)