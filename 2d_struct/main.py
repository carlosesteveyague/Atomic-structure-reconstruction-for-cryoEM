#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:58:02 2022

@author: ce423
"""

import torch

from time import time as t


from data_generator.struct2d import struct2d_gen
    
struct_features = {
    'box_side': 20,
    'hands': [25, 10],
    'arrowheads': [8,5],
    'delta': 3.5,
    
    'init_pos_hands': [-.25,.25], ## between 0 and 1 
    'init_pos_box': 0.,  ## between -1 and 1
    }
        

struct = struct2d_gen(struct_features)

from data_generator.dataset import dataset_gen

dataset_features = {
    'struct' : struct,    
    
    'n_imgs': 4000,
    'flex_var': [.5, .5, .5],  ### [hand_1, hand_2, box_shape] 
    
    'noise': 3.,
    'density_type': 'Gaussian',
    'sigma_density': 7.,
    'n_px': [64,64],
    
    'noise_CT' : 50.,
    'sigma_density_CT' : 7.,
    'n_px_CT' : 128
    }


[data_noisy, data_clean], [CT_data_noisy, CT_data_clean], diffs_data, orientations = dataset_gen(dataset_features, loop = True)


import matplotlib.pyplot as plt

idx = 0
plt.imshow(data_clean[idx].transpose(0,1).flip(0,), cmap='Greys')
plt.show()
plt.imshow(data_noisy[idx].transpose(0,1).flip(0,), cmap='Greys')
plt.show()
plt.plot(CT_data_clean[idx])
plt.show()
plt.plot(CT_data_noisy[idx])
plt.show()


#%%

from tools.graph import graph_gaussian_kernel, laplacian, eigen_graph

sigma_gauss_kernel = 0.5*max(dataset_features['n_px'])*dataset_features['noise']

start = t()
graph = graph_gaussian_kernel(data_noisy, sigma_gauss_kernel)
print('Time to compute the graph Laplacian:', t() - start)


start = t()
Lap = laplacian(graph, True)
[eigen_val, eigen_vec] = eigen_graph(Lap, 20)
print('Time to compute the eigenvectors:', t() - start)

#%%

from tools.plots import disp_point_cloud


disp_point_cloud(eigen_vec[:,[1,2,3]])
disp_point_cloud(eigen_vec[:,[1,2,4]])
disp_point_cloud(eigen_vec[:,[1,3,4]])
disp_point_cloud(eigen_vec[:,[2,3,4]])

#%%

from model.model_2d import model_2dCT

model_features = {
        'struct' : struct,
        'param_idxs' : [struct.hand1_theta_idx, 
                        struct.hand2_theta_idx, 
                        struct.box_corner_idxs[0],
                        struct.box_corner_idxs[1],
                        struct.box_corner_idxs[2],
                        struct.box_corner_idxs[3]],
        #'param_idxs' : [i for i in range(struct.Theta.shape[0])],
        'n_eigenval' : eigen_vec.shape[1],
        
        'density_type' : 'Gaussian',
        'sigma_density' : 7.,
        'n_px': [64,64],
        
        
        'sigma_density_CT' : 7.,
        'n_px_CT': 128,
        }


model = model_2dCT(model_features)


### Training

from tools.training import Dataset, train_model

train_pctg = .9
n_total = diffs_data.shape[0]
n_train = int(train_pctg*n_total)
n_test = n_total - n_train 
train_idx, test_idx = torch.utils.data.random_split(range(n_total), [n_train, n_test])


train_feat = eigen_vec[train_idx]
train_orient = orientations[train_idx]
train_inputs = torch.cat((train_feat, train_orient), dim = -1)

training_data = Dataset(train_inputs, CT_data_noisy[train_idx])


test_feat = eigen_vec[test_idx]
test_orient = orientations[test_idx]
test_inputs = torch.cat((test_feat, test_orient), dim = -1)

disc_cirve_test = struct.compute_BB(diffs_data[test_idx])

test_data_pointcloud = Dataset(test_inputs, disc_cirve_test)
test_data = Dataset(test_inputs, CT_data_clean[test_idx])




#%%


training_params = {'batch_size': 500,
          'shuffle': True,
          'max_epochs' : 3,
          'learning_rate' : .5,
          'momentum': .9}

train_model(model, training_data, test_data, test_data_pointcloud, training_params)


#%%

### Plot the images and the predictions in the test data

test_imgs_data = data_clean[test_idx]

test_CT_data = CT_data_noisy[test_idx]


test_inputs[:, -2] = 0.
test_inputs[:, -1] = -1.

test_imgs_pred = model.forward_2d(test_inputs).detach()

zero_inputs = torch.zeros(test_inputs.shape)
zero_inputs[:, -1] = -1.

init_imgs_pred = model.forward_2d(zero_inputs).detach()

for i in range(0,5):
    
    plt.subplot(1, 3, 1)
    plt.imshow(init_imgs_pred[i].transpose(0,1).flip(0,), cmap='Greys')
    plt.title('Initial prediction')
    
    plt.subplot(1, 3, 2)
    plt.imshow(test_imgs_data[i].transpose(0,1).flip(0,), cmap='Greys')
    plt.title('Ground truth')
    
    plt.subplot(1, 3, 3)
    plt.imshow(test_imgs_pred[i].transpose(0,1).flip(0,), cmap='Greys')
    plt.title('Reconstructed image')
    
    name = 'Figures_paper/Fig_2d_predictions_no_prior/volume_pred%i' %i
    #plt.savefig(name)
    plt.show()


#%%

BB_pred = model.forward_BB(test_inputs[:10]).detach()

for i in range(0,5):
    
    plt.subplot(1,2,1)
    plt.scatter(disc_cirve_test[i,:,0], disc_cirve_test[i,:,1], c = 'black')
    plt.plot(disc_cirve_test[i,:,0], disc_cirve_test[i,:,1])
    plt.xlim([-struct.image_side_lim, struct.image_side_lim])
    plt.ylim([struct.image_bottom_lim, struct.image_top_lim])
    plt.title('Ground truth')

    plt.subplot(1,2,2)
    plt.scatter(BB_pred[i,:,0], BB_pred[i,:,1], c = 'black')
    plt.plot(BB_pred[i,:,0], BB_pred[i,:,1])
    plt.xlim([-struct.image_side_lim, struct.image_side_lim])
    plt.ylim([struct.image_bottom_lim, struct.image_top_lim])
    plt.title('Prediction')
    
    name = 'Figures_paper/Fig_2d_predictions_no_prior/atomic_model_pred%i' %i
    #plt.savefig(name)
    plt.show()

