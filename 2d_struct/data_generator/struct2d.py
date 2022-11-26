#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:49:50 2022

@author: carlosesteveyague
"""

import numpy as np
import torch

from tools.particle_densities import density_comp, RadonTransform_Gaussians



class struct2d_gen:
    
    def __init__(self, clock_features):
        
        if clock_features['box_side']%2 ==1:
            raise RuntimeError('Error: side must be an even number')
        
        side = clock_features['box_side']
        hands = clock_features['hands']
        arrowheads = clock_features['arrowheads']
        
        
        init_pos_hands = clock_features['init_pos_hands']
        init_pos_box = clock_features['init_pos_box']
        
        self.dist = clock_features['delta']
        
        n_param = 5*side + sum(hands) + sum(arrowheads) - 1
        
        
        self.ref_idx = hands[0] + arrowheads[0] #+ int(2.5*side) 
        
        self.Theta = torch.zeros(n_param)
        
        
        self.hand1_theta_idx = hands[0] - 1 + arrowheads[0]
        self.hand2_theta_idx = hands[0] + 5*side - 1 + arrowheads[0]
        
        self.box_corner_idxs = [hands[0] + i*side - 1 + arrowheads[0] for i in range(1,5)]
        
        
        self.Theta[arrowheads[0] - 1] = 5*np.pi/6
        self.Theta[self.hand1_theta_idx] = 2*np.pi*init_pos_hands[0]
        self.Theta[hands[0] + int(side/2) - 1 + arrowheads[0]] = -np.pi/2
        self.Theta[self.box_corner_idxs[0]] = -np.pi/2*(1 + init_pos_box)
        self.Theta[self.box_corner_idxs[1]] = -np.pi/2*(1 - init_pos_box)
        self.Theta[self.box_corner_idxs[2]] = -np.pi/2*(1 + init_pos_box)
        self.Theta[self.box_corner_idxs[3]] = -np.pi/2*(1 - init_pos_box) 
        self.Theta[hands[0] + int(4.5*side) - 1  + arrowheads[0]] = -np.pi/2
        self.Theta[self.hand2_theta_idx] = -2*np.pi*init_pos_hands[1]
        self.Theta[hands[0] + 5*side + hands[1] - 1 + arrowheads[0]] = 7*np.pi/6
        
        
        #Limits for the images
        
        width = .8*max([hands[0],hands[1],3*side/2])
        
        self.image_top_lim = width*self.dist*1.2
        self.image_bottom_lim = -width*self.dist*1.2
        self.image_side_lim = width*self.dist*1.2

        
        
    
    def compute_BB(self, Theta_diffs, orientations = None):
        
        n = Theta_diffs.shape[0]
        
        Theta = self.Theta.unsqueeze(0).repeat(n,1)
        
        
        #Theta[:, self.hand1_theta_idx] += hand_diff[:, 0]
        #Theta[:, self.hand2_theta_idx] += -hand_diff[:, 1]
        
        Theta +=  Theta_diffs
        
        CosTheta = torch.cos(Theta)
        SinTheta = torch.sin(Theta)
        
        R = torch.zeros(CosTheta.shape[0], CosTheta.shape[1], 2, 2)
        
        R[:,:,0,0] = CosTheta
        R[:,:,0,1] = -SinTheta
        R[:,:,1,0] = SinTheta
        R[:,:,1,1] = CosTheta
        
        # Here we compute the curve from ref_idx forward
        if orientations == None:
            v0 = torch.Tensor([[0.,-1.]]).repeat(n,1)
        else:
            v0 = orientations/orientations.norm(2, dim=-1).unsqueeze(-1)
        
        V_forward = v0.unsqueeze(1)
        x0 = torch.zeros([n,1,2])
        
        for i in range(R[:, self.ref_idx :].shape[1]):
            
            r = R[:,self.ref_idx + i] 
            V_forward = torch.cat((V_forward, torch.bmm(r, V_forward[:,-1].unsqueeze(-1)).transpose(-1,-2)), dim = 1)
        
        Gamma = torch.cat((x0, torch.mul(V_forward, self.dist)), dim = 1)
        Gamma = torch.cumsum(Gamma, dim=1)
        
        # Here we compute the curve from ref_idx backward, only in the case idx_ref > 0.
        if self.ref_idx>0:
            V_backward = -torch.bmm(R[:,self.ref_idx-1].transpose(-1,-2), v0.unsqueeze(-1)).transpose(-1,-2)
            
            for i in range(R[:, :self.ref_idx-1].shape[1]):
                
                r = R[:, self.ref_idx-2-i]
                V_backward = torch.cat((V_backward, torch.bmm(r.transpose(-1,-2), V_backward[:,-1].unsqueeze(-1)).transpose(-1,-2)), dim = 1)
        
            Gamma_backward = x0 + torch.cumsum(torch.mul(V_backward, self.dist), dim=1).flip((1,))
            
            Gamma = torch.cat((Gamma_backward, Gamma), dim = 1)
            
        return Gamma
    
    def get_images_loop(self, n_px, hand_diff, noise, density_type, sigma_density, orientations = None):
        
        X = torch.linspace(-self.image_side_lim, self.image_side_lim, n_px[0])
        Y = torch.linspace(self.image_bottom_lim, self.image_top_lim, n_px[1])
        
        X,Y = torch.meshgrid(X,Y)
        
        Gamma = self.compute_BB(hand_diff, orientations)
        
        clean_imgs = torch.zeros(X.shape).unsqueeze(0).repeat(hand_diff.shape[0], 1, 1)
        
        for i in range(hand_diff.shape[0]):
            clean_imgs[i] = density_comp(X,Y,Gamma[i], sigma_density, density_type)
        
        if noise > 0:
            return clean_imgs + noise*torch.randn(clean_imgs.shape), clean_imgs
        else:
            return clean_imgs
        
    
    def get_CT_images(self, n_px, hand_diff, noise, sigma_density, orientations = None):
        
        X = torch.linspace(-self.image_side_lim, self.image_side_lim, n_px)
        
        Gamma = self.compute_BB(hand_diff, orientations)
        
        clean_imgs = RadonTransform_Gaussians(X, Gamma, sigma_density)
               
        if noise > 0:
            return clean_imgs + noise*torch.randn(clean_imgs.shape), clean_imgs
        else:
            return clean_imgs 