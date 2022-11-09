#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:52:44 2022

@author: ce423
"""

import torch
import torch.nn as nn

from tools.DFF import R_from_angles
from tools.orientation import rotation_from_params

from tools.particle_densities import density_comp



class model_chain(nn.Module):
    
    def __init__(self, model_features):
        super(model_chain, self).__init__()
        
        Psi = model_features['Psi_init']
        Theta = model_features['Theta_init']
        dists = model_features['dists']
        ref_idx = model_features['ref_idx']
        img_lims = model_features['img_lims']
        
        n_px = model_features['n_px']
        sigma_density = model_features['sigma_density']
        density_type = model_features['density_type']
        
        self.struct = chain_batch_model(Psi, Theta, dists, ref_idx, img_lims, n_px, sigma_density, density_type, model_features['CTF_feat'])
        
        self.param_idxs = model_features['param_idxs']
        n_eigenval = model_features['n_eigenval']
        
        self.GammaPsi = nn.Linear(n_eigenval, len(self.param_idxs), bias = False)
        self.GammaTheta = nn.Linear(n_eigenval, len(self.param_idxs), bias = False)
        
        self.GammaPsi.weight = nn.Parameter(torch.zeros([len(self.param_idxs), n_eigenval]))
        self.GammaTheta.weight = nn.Parameter(torch.zeros([len(self.param_idxs), n_eigenval]))
        
    def forward(self, inputs):
        
        V = inputs[:, :-7]
        orientations = inputs[:, -7:-4]
        x0s = inputs[:,-4:-1]
        Df = inputs[:,-1]
        
        Psi_diffs = torch.zeros(V.shape[0], self.struct.Psi.shape[1])
        Theta_diffs = torch.zeros(V.shape[0], self.struct.Theta.shape[1])
        
        Psi_diffs[:, self.param_idxs] = self.GammaPsi(V)
        Theta_diffs[:, self.param_idxs] = self.GammaTheta(V)
        
        return self.struct.CT_images(Psi_diffs, Theta_diffs, orientations, x0s, Df)
    
    def forward_disc_curve(self, inputs):
        
        V = inputs[:, :-7]
        orientations = inputs[:, -7:-4]
        x0s = inputs[:,-4:-1]
        
        Psi_diffs = torch.zeros(V.shape[0], self.struct.Psi.shape[1])
        Theta_diffs = torch.zeros(V.shape[0], self.struct.Theta.shape[1])
        
        Psi_diffs[:, self.param_idxs] = self.GammaPsi(V)
        Theta_diffs[:, self.param_idxs] = self.GammaTheta(V)
        
        return self.struct.discrete_curve(Psi_diffs, Theta_diffs, orientations, x0s)
        
    def get_diffs(self, V):
        
        Psi_diffs = torch.zeros(V.shape[0], self.struct.Psi.shape[1])
        Theta_diffs = torch.zeros(V.shape[0], self.struct.Theta.shape[1])
        
        Psi_diffs[:, self.param_idxs] = self.GammaPsi(V)
        Theta_diffs[:, self.param_idxs] = self.GammaTheta(V)
        
        return torch.cat([Psi_diffs, Theta_diffs], dim = 1)
    
    def pred_images_without_CTF(self, inputs):
        
        V = inputs[:, :-6]
        orientations = inputs[:, -6:-3]
        x0s = inputs[:,-3:]
                
        Psi_diffs = torch.zeros(V.shape[0], self.struct.Psi.shape[1])
        Theta_diffs = torch.zeros(V.shape[0], self.struct.Theta.shape[1])
        
        Psi_diffs[:, self.param_idxs] = self.GammaPsi(V)
        Theta_diffs[:, self.param_idxs] = self.GammaTheta(V)
        
        return self.struct.CT_images(Psi_diffs, Theta_diffs, orientations, x0s)
        

from tools.CTF import CTF_filters  

from tools.fft import fft2_center, ifft2_center

class chain_batch_model:
    
    
    def __init__(self, Psi, Theta, dists, ref_idx, img_lims, n_px, sigma_density, density_type, CTF_feat):
        
        self.ref_idx = ref_idx
        
        self.dists = dists.unsqueeze(0)
        
        self.Psi = Psi.unsqueeze(0)
        self.Theta = Theta.unsqueeze(0)
        
        self.x_limsup = img_lims[0]
        self.x_liminf = img_lims[1]
        self.y_limsup = img_lims[2]
        self.y_liminf = img_lims[3]
        
        self.n_px = n_px
        self.sigma_density = sigma_density
        self.density_type = density_type
        
        self.CTF_feat = CTF_feat
    
    def discrete_curve(self, Psi_diff, Theta_diff, orientations, x0s):
        """ 
        This function calculates the discrete curve associated to each set of parameters in the batch.
        
        Psi_diff and Theta_diff represents the perturbation of the rotation angles with respect to the given conformation.
        
        """
        
        F0 = rotation_from_params(orientations)  
        x0 = x0s.unsqueeze(1)

        
        # We compute the transition matrices from the Euler angles.
        R = self.compute_R(Psi_diff, Theta_diff)
        
        # Here we compute the curve from ref_idx forward
        DFF_forward = F0.unsqueeze(1)   
        
        for i in range(R[:,self.ref_idx :].shape[1]):
            
            r = R[:, self.ref_idx + i] 
            DFF_forward = torch.cat((DFF_forward, torch.bmm(r, DFF_forward[:,-1]).unsqueeze(1)), dim = 1)
        
        Gamma = torch.cat((x0, torch.mul(DFF_forward[:,:,2], self.dists[:, self.ref_idx:, :])), dim = 1)
        Gamma = torch.cumsum(Gamma, dim=1)
        
        # Here we compute the curve from ref_idx backward, only in the case idx_ref > 0.
        if self.ref_idx>0:
            DFF_backward = -torch.bmm(R[:, self.ref_idx-1].transpose(-1,-2), F0).unsqueeze(1)
            
            for i in range(R[:,:self.ref_idx-1].shape[1]):
                
                r = R[:, self.ref_idx-2-i]
                DFF_backward = torch.cat((DFF_backward, torch.bmm(r.transpose(-1,-2), DFF_backward[:,-1]).unsqueeze(1)), dim = 1)
        
            Gamma_backward = x0 + torch.cumsum(torch.mul(DFF_backward[:,:,2], self.dists[:, :self.ref_idx, :].flip((1,))), dim=1).flip((1,))
            
            Gamma = torch.cat((Gamma_backward, Gamma), dim = 1)
            
        return Gamma
    
    def compute_R(self, Psi_diff, Theta_diff):
        Psi = self.Psi + Psi_diff
        Theta = self.Theta + Theta_diff
        return R_from_angles(Psi, Theta)   
    
    def CT_images(self, Psi_diff, Theta_diff, orientations, x0s, Df = None):
        
        curve_proj = self.discrete_curve(Psi_diff, Theta_diff, orientations, x0s)[:,:,:-1]
        
        X = torch.linspace(self.x_liminf, self.x_limsup, self.n_px)
        Y = torch.linspace(self.y_liminf, self.y_limsup, self.n_px)
        
        side = X[-1] - X[0]
        
        X,Y = torch.meshgrid(X,Y)
        
        clean_imgs = torch.zeros(X.shape).unsqueeze(0).repeat(curve_proj.shape[0], 1, 1)
        
        for i in range(curve_proj.shape[0]):
            clean_imgs[i] = density_comp(X,Y,curve_proj[i], self.sigma_density, self.density_type)
        
        if Df != None:
            filters = CTF_filters(clean_imgs.shape[-1], side, self.CTF_feat['kV'], Df, self.CTF_feat['Cs'], 
                                  self.CTF_feat['alpha'], self.CTF_feat['B_factor'], self.CTF_feat['phase_shift'])
        
            f_clean_imgs = fft2_center(clean_imgs)
        
            CTF_clean_imgs = filters*f_clean_imgs
        
            clean_imgs = ifft2_center(CTF_clean_imgs).abs()
        
        return clean_imgs 