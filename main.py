# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:05:21 2022

@author: Dell
"""
import sys
sys.path.append('D:/NCSU Subjects/NCSU Sem 2/ECE 763/project03/Final Code/')
import os
import argparse
from Utils.LoadDataset import *
from Training.train_model import Train
from Utils.tools import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Models.AlexNet import AlexNet


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training script"
    )

    parser.add_argument("model",  help="Model to use for training",nargs = '?',default = "AlexNet")
    parser.add_argument("optimizer", help="optimizer for the model",nargs = '?',default = "SGD")
    parser.add_argument("device", help="Device to train the model on",nargs = '?',default = "cpu")
    parser.add_argument("outputs", help="No of outputs", type=float,nargs = '?',default = "2")
    parser.add_argument("epochs", help="No of epochs", type=float,nargs = '?',default = "20")
    parser.add_argument("dataset", help="Dataset to use for training",nargs = '?',default = "FDDB")
    parser.add_argument("directory", help="Directory of the dataset",nargs = '?',default = "D:/NCSU Subjects/NCSU Sem 2/ECE 763/project03/Final Code/Dataset/")


    args = parser.parse_args()

    dataset = LoadDataset(args.dataset, args.directory)
    
    traintiny, validtiny, trainset, testset, validset = dataset.load_dataset()
    
    #Printing first loss
    tr = Train(args.model,args.optimizer,args.dataset)
    tr.train_model(trainset, validset, args.device, args.optimizer, 1, 10**-3, 0)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)
    first_loss, _ = evaluate(tr.model, testset, criterion, args.device)
    print(f'first Loss: {first_loss}')
    
    #Overfitting Model
    tr = Train(args.model,args.optimizer,args.dataset)
    # tr.train_model(traintiny, validtiny, args.device, args.optimizer, 40,10**-3, 0)
    
    # #Train loss and validation loss barely change with low learning rate
    # tr.train_model(trainset, validset, args.device, args.optimizer, 10,10**-6, 10**-6)
    # tr.train_model(trainset, validset, args.device, args.optimizer, 10,10**-3, 10**-6)
    """
    Using these two blocks to get the optimal learning rate values
    Updating these values in the train_model function below
    
    #Coarse Train
    max_count = 20

    for count in tqdm(range(max_count)):
        LR = 10**random.uniform(-5,5)
        reg = 10**random.uniform(-3,-6)
        tr.train_model(trainset, validset, args.device, args.optimizer, 10,LR, reg)
        
    # #Fine Train
    max_count = 10

    for count in tqdm(range(max_count)):
        LR = 10**random.uniform(-4,-1)
        reg = 10**random.uniform(-3,-5)
        tr.train_model(trainset, validset, args.device, args.optimizer, 10,LR, reg)
        
    """
    tr.train_model(trainset, validset, args.device, args.optimizer, 20,0.0012, 0.001)    
    test_loss, test_acc = evaluate(tr.model, testset, criterion, args.device)
    print(f'Test Accuracy: {test_acc} | Test Loss: {test_loss}')