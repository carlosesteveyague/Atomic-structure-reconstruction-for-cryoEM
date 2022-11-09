#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 18:29:24 2022

@author: carlosesteveyague
"""

import torch
from scipy.linalg import logm

A_1 = torch.tensor([[[0.,1.,0.],[0.,0.,0.],[0.,0.,1.]]], dtype = torch.float)
A_2 = torch.tensor([[[-1.,0.,0.],[0.,0.,1.],[0.,0.,0.]]], dtype = torch.float)
A_3 = torch.tensor([[[0.,0.,0.],[0.,-1.,0.],[-1.,0.,0.]]], dtype = torch.float)

A = torch.cat((A_1, A_2, A_3), dim=0)

def rotation_from_params(params):
    """
    Given a vector with three paramters 
    
    params = [kappa, tau, eta]
    
    this function computes the element of SO(3) given by
    
    exp(A), 
    
    where A is the antisymmetric matrix
    
    A = [[0,  kappa, eta],
         [-kappa, 0 , tau],
         [-eta, -tau , 0]]

    Parameters
    ----------
    params : TYPE torch.float tensor
        DESCRIPTION. 3 parameters defining an element in the Lie algebra associated to SO(3)

    Returns
    -------
    TYPE torch.float tensor
        DESCRIPTION. 3x3 matrix. An element of SO(3)

    """
    A_params = torch.matmul(params, A).transpose(0,1)
    return torch.matrix_exp(A_params)


def params_from_rotation(R):
    """
    This function is the inverse of the function rotation_from_params.
    
    Given a rotation matrix R, this function computes the parameters
    [kappa, tau, eta] defining an element in the Lie algebra associated to SO(3).
    """
    
    params = torch.empty([R.shape[0],3], dtype = torch.float)
    
    for i in range(R.shape[0]):
        A_aux, err = logm(R[i].detach(), False)
        if err > 1e-4:
            print('Warning: error computing logm is large:', err)
        kappa = A_aux[0,1].real
        tau = A_aux[1,2].real
        eta = A_aux[0,2].real
        params[i] = torch.tensor([kappa, tau, eta])
           
    return params
    
    