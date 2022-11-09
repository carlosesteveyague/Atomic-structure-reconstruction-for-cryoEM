#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:30:39 2022

@author: ce423
"""

from torch.fft import fft2, fftshift, ifft2, ifftshift


def fft2_center(img):
    img = fftshift(img, dim = (-1,-2))
    img = fft2(img)
    img = fftshift(img) 
    return img

def ifft2_center(img):
    img = ifftshift(img, dim = (-1,-2))
    img = ifft2(img)
    img = ifftshift(img) 
    return img