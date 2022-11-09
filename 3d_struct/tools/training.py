#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:22:38 2022

@author: ce423
"""

import torch

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, eigen_vecs, labels):
        'Initialization'
        self.labels = labels
        self.eigen_vecs = eigen_vecs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.eigen_vecs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.eigen_vecs[index]

        y = self.labels[index]

        return X, y




import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from tqdm import tqdm

from time import time

def train_model(model, training_data, test_data, test_data_pointcloud, training_params):
    
    max_epochs = training_params['max_epochs']
    lr = training_params['learning_rate']
    momentum = training_params['momentum']
    
    start = time()
    
    # Generator
    params = {
            'batch_size': training_params['batch_size'],
            'shuffle': training_params['shuffle']
            }
                
    training_generator = torch.utils.data.DataLoader(training_data, **params)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum)
    
    Training_loss_history = torch.zeros(max_epochs)
    Test_loss_history = torch.zeros(max_epochs+1)
    Test_avg_pointcloud_history = torch.zeros(max_epochs+1)
    Test_max_pointcloud_history = torch.zeros(max_epochs+1)
    Test_max_error_history = torch.zeros(max_epochs+1)
    
    with torch.no_grad():
        
        test_input_local, test_local_data = test_data[:]
        
        outputs = model(test_input_local)
        loss = criterion(outputs, test_local_data)
        print('Initial test loss:', loss.item())
        Test_loss_history[0] = loss.item()
        
        test_input_local, test_BB_local = test_data_pointcloud[:]
        
        point_cloud_error = (model.forward_disc_curve(test_input_local) - test_BB_local).norm(dim = -1)
        error_avg = point_cloud_error.mean()
        print('Initial test average point cloud error:', error_avg.item())
        Test_avg_pointcloud_history[0] = error_avg.item()
        
        error_max_avg = point_cloud_error.max(-1)[0].mean()
        print('Initial test average max point cloud error:', error_max_avg.item())
        Test_max_pointcloud_history[0] = error_max_avg.item()
        
        error_max = point_cloud_error.max()
        print('Initial test max point cloud error:', error_max.item())
        Test_max_error_history[0] = error_max.item()
    
    
    for epoch in range(max_epochs):
        # Training
        trainLoss = 0.
        samples = 0
        for local_feat, local_data in tqdm(training_generator):
            
            optimizer.zero_grad()
            
            outputs = model(local_feat)
            
            loss = criterion(outputs, local_data)
            
            loss.backward()
            optimizer.step()
            
            trainLoss += loss.item()*local_feat.size(0)
            samples += local_feat.size(0)
        
                    
        print('Epoch', epoch+1, 'completed.')
        print('Training loss:', trainLoss/samples)
        Training_loss_history[epoch] = trainLoss/samples
        
        if epoch%2 >=0:
            with torch.no_grad():
                
                test_input_local, test_local_data = test_data[:]
                
                outputs = model(test_input_local)
                loss = criterion(outputs, test_local_data)
                print('Test loss:', loss.item())
                Test_loss_history[epoch+1] = loss.item()
                
                test_input_local, test_BB_local = test_data_pointcloud[:]
                
                point_cloud_error = (model.forward_disc_curve(test_input_local) - test_BB_local).norm(dim = -1)
                error_avg = point_cloud_error.mean()
                print('Test average point cloud error:', error_avg.item())
                Test_avg_pointcloud_history[epoch+1] = error_avg.item()
                
                error_max_avg = point_cloud_error.max(-1)[0].mean()
                print('Test average max point cloud error:', error_max_avg.item())
                Test_max_pointcloud_history[epoch+1] = error_max_avg.item()
                
                error_max = point_cloud_error.max()
                print('Test  max point cloud error:', error_max.item())
                Test_max_error_history[epoch+1] = error_max.item()
                
    print('Training finished. Elapsed time:',time() - start)
    
    
    plt.plot(Training_loss_history)
    plt.title('Training Loss History')
    #plt.savefig('Figures_paper/Fig_training_3d/training_loss')
    plt.show()
    
    plt.plot(Test_loss_history)
    plt.title('Test Loss History')
    #plt.savefig('Figures_paper/Fig_training_3d/test_loss')
    plt.show()
    
    plt.plot(Test_avg_pointcloud_history)
    plt.plot(Test_max_pointcloud_history)
    plt.title('Test average point cloud error')
    plt.ylim(0)
    #plt.savefig('Figures_paper/Fig_training_3d/pointcloud_error')
    plt.show()
    
    print('Training finished. Elapsed time:',time() - start)
    print('Training loss:', trainLoss/samples)
    print('Test loss:', loss.item())
    print('Test average point cloud error:', error_avg.item())
    print('Test average max point cloud error:', error_max_avg.item())
    print('Test  max point cloud error:', error_max.item())