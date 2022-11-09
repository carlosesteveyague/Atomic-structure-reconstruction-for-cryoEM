#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:37:52 2022

@author: ce423
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
    