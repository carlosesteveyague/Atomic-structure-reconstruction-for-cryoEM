#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:31:15 2022

@author: ce423
"""

import torch
import numpy as np


def voltage_to_wavelength(voltage):
    Lambda = 12.2643247/np.sqrt(voltage*1e+3 + 0.978466*voltage**2);
    return Lambda


def CTF(xi_norm_sq , voltage, Df, C_s, alpha, B_factor, phase_shift):
    
    phase_shift = phase_shift*np.pi/180 
    C_s = C_s*1e+7 # convert mm to Angstroms
    Df = Df.unsqueeze(-1).unsqueeze(-1)*1e+3 # convert to Angstroms
    xi_norm_sq = xi_norm_sq.repeat(Df.shape[0], 1, 1)
    
    Lambda = voltage_to_wavelength(voltage)
    
    Theta = 2*np.pi*(-.5*Lambda*Df*xi_norm_sq + .25*C_s*Lambda**3*xi_norm_sq**2) - phase_shift
    
    Env = torch.exp(-.25*B_factor*xi_norm_sq) 
    
    return (np.sqrt(1-alpha**2)*torch.sin(Theta) - alpha*torch.cos(Theta))*Env


def CTF_filters(n_px, img_side , voltage, Df, C_s, alpha, B_factor, phase_shift):
    X = np.linspace(-.5, .5, n_px, endpoint = False)*n_px/img_side
    X,Y = np.meshgrid(X,X)
    
    xi_norm_sq = torch.tensor(np.expand_dims(X**2 + Y**2, axis = 0))
    
    return CTF(xi_norm_sq , voltage, Df, C_s, alpha, B_factor, phase_shift)