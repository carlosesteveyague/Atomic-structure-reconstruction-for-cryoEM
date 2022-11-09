#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:02:38 2022

@author: ce423
"""

import torch

from tools.DFF import R_from_angles
from tools.orientation import rotation_from_params

from tools.particle_densities import density_comp

from data_generator.Imaging import image_CT, CTF_image_CT


from tqdm import tqdm

class chain_structure:
    
    """
    This class contains the parametrization of a protein backbone.
    
    The parameters are the following:
        - Euler angles, 
        - the distances between consecutive atoms in the backbone
        - a reference point in the curve (ref_idx), 
        - and a reference orientation.
    
    It has, as an attribute, the index of the reference atom in the backbone. By default we take
    the first atom, but it is more convenient to use an atom in the middle of the backbone.
    
    The function forward calculates the curve associated to the parametrization.
    """
    
    def __init__(self, Psi, Theta, x0, Orientation, dists, ref_idx):
        
        self.ref_idx = ref_idx
        
        self.x0 = x0
        
        self.orientation = Orientation
        
        self.dists = dists
        
        self.Psi = Psi
        self.Theta = Theta
    
    def discrete_curves(self, IniCond_diff = None):
        """ 
        This function calculates the discrete curve associated to each set of parameters in the batch.
        
        IniCond_diff allows to rotate the curves with respect to the original orientation.
        
        """
        
        if IniCond_diff == None:
            F0 = rotation_from_params(self.orientation)  
            x0 = self.x0.unsqueeze(1)
        else:
            F0 = rotation_from_params(self.orientation + IniCond_diff)
            x0 = self.x0.unsqueeze(1)
        
        # We compute the transition matrices from the Euler angles.
        R = self.compute_R()
        
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
    
    def compute_R(self):
        return R_from_angles(self.Psi, self.Theta)
    
    
    def projected_curves(self, orient_diffs = None):        
        return self.discrete_curves(orient_diffs)[:,:,:-1]
    
    def CT_images(self, orient_diffs, n_px, noise, density_type, sigma_density, CTF_data_feat = None):
        
        if CTF_data_feat == None:
            Df = None
            if noise > 0:
                noisy_imgs, clean_imgs, img_lims = image_CT(self, orient_diffs, n_px, noise, density_type, sigma_density)
                return noisy_imgs, clean_imgs, clean_imgs, img_lims, Df
            else:
                clean_imgs, img_lims = image_CT(self, orient_diffs, n_px, 0, density_type, sigma_density)
                return clean_imgs, clean_imgs, img_lims, Df
        else:
            if noise > 0:
                noisy_imgs, clean_imgs, clean_imgs_no_CTF, img_lims, Df = CTF_image_CT(self, orient_diffs, n_px, noise, density_type, sigma_density, CTF_data_feat)
                return noisy_imgs, clean_imgs, clean_imgs_no_CTF, img_lims, Df
            else:
                clean_imgs, clean_imgs_no_CTF, img_lims, Df = CTF_image_CT(self, orient_diffs, n_px, 0, density_type, sigma_density, CTF_data_feat)
                return clean_imgs, clean_imgs_no_CTF, img_lims, Df