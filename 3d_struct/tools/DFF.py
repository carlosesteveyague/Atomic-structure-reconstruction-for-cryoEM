#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:35:59 2022

@author: ce423
"""

import torch

from tools.orientation import params_from_rotation

def compute_DFF(X):
    """
    This function computes the Discrete Frenet Frames for each of the discrete curves in a batch of discrete curves.
    
    
    The frist output is a tensor with the N-1 Frenet Frames (rotation matrices), 
    whiere M is the number of points in the sequence.
    
    The third row in each frame corresponds to the unitary vector in the direction x_{i+1} - x_i
    
    The second output is a 1-dimensional tensor with the distances between every two consecutive
    points in the sequence.

    Parameters
    ----------
    X : TYPE  torch.float tensor
        DESCRIPTION. 3-d tensor with shape [N,L,3]
        
        N is the batch size,
        L is the length of each discrete curve

    Returns
    -------
    DFF : TYPE  torch.float tensor
        DESCRIPTION. 4-d tensor with shape [N,L,3,3]
        
        N sequences of L-1 orthonormal 3x3 matrices. 
        The third row in each matrix corresponds to the direction (X_{i+1} - x_i)/dist_i
        
    dists : TYPE torch.float tensor 
        DESCRIPTION. 2-d tensor with shape [N,L]
        N sequences of L-1 positive reals given by |X_{i+1} - X_i|
            
    """
    
    T = X[:, 1:] - X[:, :-1]
    dists = T.norm(dim = -1).unsqueeze(-1)
    T = T/dists
    
    B = torch.cross(T[:,:-1], T[:,1:], dim = -1)
    B = B/B.norm(dim = -1).unsqueeze(-1)
    
    B = torch.cat((B[:,0].unsqueeze(1), B), dim = 1)
    
    N = torch.cross(B,T, dim = -1)
    
    DFF = torch.cat((N.unsqueeze(-2), B.unsqueeze(-2), T.unsqueeze(-2)), dim = -2)
    
    return DFF, dists

def compute_parameters(X, ref_idx = 0):
    """
    This function computes the parameters for a given batch of discrete curves with the same length.
    The inputs are the batch of point clouds and the reference index, i.e. the starting point of the curves 
    (the same for all the curves in X).

    Parameters
    ----------
    X : TYPE torch tensor with shape [N,L,3]
        DESCRIPTION. N is the size of the batch (number of discrete curves)
                    L is the length of the discrete curves.
    ref_idx : TYPE intiger between 0 and L-1
        DESCRIPTION.

    Returns
    -------
    Psi : TYPE torch tensor with shape [N, L]
        DESCRIPTION.
    Theta : TYPE torch tensor with shape [N, L]
        DESCRIPTION.
    x0 : TYPE torch tensor with shape [N, 3]
        DESCRIPTION. position of the reference atom for each discrete curve.
    Orientation : TYPE torch tensor with shape [N, 3]
        DESCRIPTION. parameters of the orientation for each backbone
    dist : TYPE float
        DESCRIPTION. distance between consecutive atoms in the curve

    """
    
    DFF, dists = compute_DFF(X)
    R = Rotations_from_DFF(DFF)
    Psi, Theta = Angles_from_R(R)
    
    
    #dist = dists.mean()
    
    x0 = X[:, ref_idx]
    
    Orientation = params_from_rotation(DFF[:, ref_idx])
    
    return Psi, Theta, x0, Orientation, dists



def Rotations_from_DFF(DFF): 
    """
    Given a sequence of L-1 Frenet Frames F_i, this function computes the L-2 transition matrices,
    i.e.
    R_i = F_i F_{i-1}^{-1},  for i = 1,2,...
    
    
    The input is a batch of DFF sequences, i.e.
    
    DFF must be a torch.tensor with shape [N,L,3,3]
    """
    return torch.matmul(DFF[:,1:], DFF[:,:-1].transpose(2,3))

def Angles_from_R(R):
    """
    This function computes the two angles associated to each of the rotation matrices
    in the sequence of rotation matrices R.
    
    The rotation matrices have the form
    
    R_i = [[cos(psi_i)*cos(theta_i), cos(psi_i)*sin(theta_i), -sin(psi_i)],
           [-sin(theta_i),         cos(theta_i),          0],
           [sin(psi_i)*cos(theta_i), sin(psi_i)*sin(theta_i), cos(psi_i)]]
    
    Given the sequence of rotations R_i, this function computes the sequence of angles psi_i and theta_i.
    
    This implementation takes batches as input, i.e.
    
    R must be a tensor with shape [N,L,3,3]
    
    N is the batch size
    L is the length of the backbones
    
    The output are two tensors with shape [N, L] each.

    """
    
    R[(R>1.)*(R<1.+1e-4)] = 1.
    R[(R<-1.)*(R>-1.-1e-4)] = -1.
    
    CosPsi = R[:,:,2,2]
    SinPsi = -R[:,:,0,2]
    
    CosTheta = R[:,:,1,1]
    SinTheta = -R[:,:,1,0]
    
    Psi = torch.acos(CosPsi)
    Theta = torch.acos(CosTheta)
    
    checkPsi = torch.mul(SinPsi, torch.sin(Psi))
    checkTheta = torch.mul(SinTheta, torch.sin(Theta))
    
    Psi[checkPsi<0] = -Psi[checkPsi<0]
    Theta[checkTheta<0] = -Theta[checkTheta<0]
    
    return Psi, Theta

def R_from_angles(Psi, Theta):
    """
    Given two sequences of angles psi_i and theta_i, this function computes 
    the corresponding rotation matrices of the form
    
    R_i = [[cos(psi_i)*cos(theta_i), cos(psi_i)*sin(theta_i), -sin(psi_i)],
           [-sin(theta_i),         cos(theta_i),          0],
           [sin(psi_i)*cos(theta_i), sin(psi_i)*sin(theta_i), cos(psi_i)]]
    
    This implementation takes batches of parameters,i.e.
    
    Psi and Theta must be tensors with shape [N, L]
    
    N is the batch size and L is the length of the backbone
    
    The output is a a tensor with shape [N,L,3,3]
    
    """
    
    CosPsi = torch.cos(Psi)
    SinPsi = torch.sin(Psi)
    
    CosTheta = torch.cos(Theta)
    SinTheta = torch.sin(Theta)
    
    R = torch.zeros([Psi.shape[0], Psi.shape[1], 3, 3], dtype = torch.float)
    
    R[:,:,0,0] = torch.mul(CosPsi, CosTheta)
    R[:,:,0,1] = torch.mul(CosPsi, SinTheta)
    R[:,:,0,2] = -SinPsi
    
    R[:,:,1,0] = -SinTheta
    R[:,:,1,1] = CosTheta
    
    R[:,:,2,0] = torch.mul(SinPsi, CosTheta)
    R[:,:,2,1] = torch.mul(SinPsi, SinTheta)
    R[:,:,2,2] = CosPsi

    return R

