#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 11:46:49 2022

@author: ce423
"""

 
import torch
from tqdm import tqdm

from tools.particle_densities import density_comp

from tools.CTF import CTF_filters

from tools.fft import fft2_center, ifft2_center


def image_CT(struct, orient_diffs, n_px, noise, density_type, sigma_density):
        
        curve_proj = struct.projected_curves(orient_diffs).detach()
        
        # Here set the limits of the images:
        # First we compute the range of the projected particles.
        x_limsup = curve_proj[:,:,0].max()
        x_liminf = curve_proj[:,:,0].min()
        y_limsup = curve_proj[:,:,1].max()
        y_liminf = curve_proj[:,:,1].min()
        
        # Then we set the centre of the image.
        x_midpoint = (x_limsup + x_liminf)/2
        y_midpoint = (y_limsup + y_liminf)/2
        
        # The length of the image size is 10 percent longer than the range of the particles.
        # We avoid the particles touching the edges of the image.
        side = 1.1*max(x_limsup - x_liminf, y_limsup - y_liminf)
        
        x_limsup = x_midpoint + side/2
        x_liminf = x_midpoint - side/2
        y_limsup = y_midpoint + side/2
        y_liminf = y_midpoint - side/2
        
        
        img_lims = torch.tensor([x_limsup, x_liminf, y_limsup, y_liminf])
        
        # We generate a rectangular grid on the image domain.
        X = torch.linspace(x_liminf, x_limsup, n_px)
        Y = torch.linspace(y_liminf, y_limsup, n_px)
        
        X,Y = torch.meshgrid(X,Y)
        
        # We create a tensor with zeros where we shall store the images,
        clean_imgs = torch.zeros(X.shape).unsqueeze(0).repeat(curve_proj.shape[0], 1, 1)
        
        # and add the images one by one by convolving each point cloud with a Gaussian kernel.
        for i in tqdm(range(curve_proj.shape[0])):
            clean_imgs[i] = density_comp(X,Y,curve_proj[i], sigma_density, density_type)
        
        
        if noise > 0:
            # here we add Gaussian noise to the clean images.
            return clean_imgs + noise*torch.randn(clean_imgs.shape), clean_imgs, img_lims
        else:
            return clean_imgs, img_lims 

def CTF_image_CT(struct, orient_diffs, n_px, noise, density_type, sigma_density,CTF_data_feat):
        
        curve_proj = struct.projected_curves(orient_diffs).detach()
        
        # Here set the limits of the images:
        # First we compute the range of the projected particles.
        x_limsup = curve_proj[:,:,0].max()
        x_liminf = curve_proj[:,:,0].min()
        y_limsup = curve_proj[:,:,1].max()
        y_liminf = curve_proj[:,:,1].min()
        
        # Then we set the centre of the image.
        x_midpoint = (x_limsup + x_liminf)/2
        y_midpoint = (y_limsup + y_liminf)/2
        
        # The length of the image size is 10 percent longer than the range of the particles.
        # We avoid the particles touching the edges of the image.
        side = 1.1*max(x_limsup - x_liminf, y_limsup - y_liminf)
        
        x_limsup = x_midpoint + side/2
        x_liminf = x_midpoint - side/2
        y_limsup = y_midpoint + side/2
        y_liminf = y_midpoint - side/2
        
        img_lims = torch.tensor([x_limsup, x_liminf, y_limsup, y_liminf])
        
        # We generate a rectangular grid on the image domain.
        X = torch.linspace(x_liminf, x_limsup, n_px)
        Y = torch.linspace(y_liminf, y_limsup, n_px)
        
        X,Y = torch.meshgrid(X,Y)
        
        # We create a tensor with zeros where we shall store the images,
        clean_imgs_no_CTF = torch.zeros(X.shape).unsqueeze(0).repeat(curve_proj.shape[0], 1, 1)
        
        # and add the images one by one by convolving each point cloud with a Gaussian kernel.
        for i in tqdm(range(curve_proj.shape[0])):
            clean_imgs_no_CTF[i] = density_comp(X,Y,curve_proj[i], sigma_density, density_type)
                
        """
        CTF below
        """
        
        Df_rand_choice = torch.randint(CTF_data_feat['Df'].shape[0], [clean_imgs_no_CTF.shape[0]])
        Df = CTF_data_feat['Df'][Df_rand_choice]
        
        filters = CTF_filters(n_px, side, CTF_data_feat['kV'], Df, CTF_data_feat['Cs'], 
                              CTF_data_feat['alpha'], CTF_data_feat['B_factor'], CTF_data_feat['phase_shift'])
        
        f_clean_imgs = fft2_center(clean_imgs_no_CTF)
        
        CTF_clean_imgs = filters*f_clean_imgs
        
        clean_imgs = ifft2_center(CTF_clean_imgs).abs()
        
        if noise > 0:
            # here we add Gaussian noise to the clean images.
            return clean_imgs + noise*torch.randn(clean_imgs.shape), clean_imgs, clean_imgs_no_CTF, img_lims, Df
        else:
            return clean_imgs, img_lims, Df