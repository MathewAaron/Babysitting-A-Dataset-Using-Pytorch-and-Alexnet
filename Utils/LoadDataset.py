# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import sys
import torchvision
from torchvision import transforms, datasets

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import random
import time
from itertools import chain
import os
from os import path
from tqdm import tqdm

def check_for_GPU():
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu') 
  print("Using:",device)
  
check_for_GPU()


def seed():
  SEED = 100
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

seed()

class LoadDataset():
    
    def __init__(self,dataset,directory):
        self.dataset = dataset
        self.directory = directory
        self.train_transform = None
        self.test_transform = None
    
    def load_dataset(self,dataset= None):
        # Using mean and std deviation
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Using Augmentation and Normalization on the dataset
        self.train_transform = transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ColorJitter(brightness=2),
                                      transforms.Resize((32,32)),
                                      transforms.ToTensor(),
                                      normalization,
                                                ])
     
        self.test_transform = transforms.Compose([
                                    transforms.Resize((32,32)),
                                    
                                     transforms.ToTensor(),
                                     normalization,
                                               ])
        trainset = []
        testset = []
        if self.dataset == 'CIFAR10':
            
            trainset = torchvision.datasets.CIFAR10(root = self.directory+"Cifar/cifar-10-python", train=True,
                                                          download=True, transform=self.train_transform)
                
            testset = torchvision.datasets.CIFAR10(root = self.directory+"Cifar/cifar-10-python", train=False,
                                                         download=True, transform=self.test_transform)
            
        elif self.dataset == 'FDDB':

                TRAIN_PATH = self.directory+'/training/'
                TEST_PATH = self.directory+'/testing/'
                trainset = datasets.ImageFolder(root= TRAIN_PATH, transform=self.train_transform)
                print(trainset)
                testset = datasets.ImageFolder(root= TEST_PATH, transform=self.test_transform)
                print(testset)

        else:
            print('Enter valid dataset name')
            
        VAL_SPLIT = 0.9 # 90%- Train, 10%- Validation

        train_examples = int(len(trainset) * VAL_SPLIT)
        valid_examples = len(trainset) - train_examples
        train_data_, valid_data_ = data.random_split(trainset, 
                                          [train_examples,valid_examples])

        
        print(f'Complete data')
        print(f'Images in training data: {len(train_data_)}')
        print(f'Images in validation data: {len(valid_data_)}')
        print(f'Images in test data: {len(testset)}')

        
        valid_data_ = copy.deepcopy(valid_data_) # create a deep copy to avoid changing main object
        valid_data_.dataset.transform = self.test_transform # apply test transforms to have consistency with test data
        
        
        BATCH_SIZE = 32 # data batch_size
        
        train_data = data.DataLoader(train_data_, 
                                        shuffle = True, 
                                        batch_size = BATCH_SIZE)
        
        valid_data = data.DataLoader(valid_data_, 
                                        batch_size = BATCH_SIZE)
        
        test_data = data.DataLoader(testset, 
                                        batch_size = BATCH_SIZE)
        
        indices = list(range(0, int(len(trainset)*0.1), 1))
        data_subset = torch.utils.data.Subset(trainset, indices)
        VAL_SPLIT = 0.5 
        train_examples = int(len(data_subset) * VAL_SPLIT)
        valid_examples = len(data_subset) - train_examples
        
        train_data_subset, valid_data_subset = data.random_split(data_subset, 
                                                  [train_examples,valid_examples])
        train_data_tiny = data.DataLoader(train_data_subset, 
                                        shuffle = True, 
                                        batch_size = BATCH_SIZE)
        
        valid_data_tiny = data.DataLoader(valid_data_subset, 
                                        batch_size = BATCH_SIZE)
       
        print(f'Batch Size: {BATCH_SIZE}')
        print(f'Batches in overfitting training data: {len(train_data_tiny)}')
        print(f'Batches in overfitting valid data: {len(valid_data_tiny)}')
        print(f'Batches in training data: {len(train_data)}')
        print(f'Batches in validation data: {len(valid_data)}')
        print(f'Batches in test data: {len(test_data)}')

            
        return train_data_tiny, valid_data_tiny, train_data, test_data, valid_data
            
            



