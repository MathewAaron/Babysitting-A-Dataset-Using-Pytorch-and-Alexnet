# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 17:16:06 2022

@author: Dell 
"""
import sys
from Models.AlexNet import AlexNet
from Utils.tools import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm

class Train:
 
  def __init__(self,model,optimizer,dataset):
        self.dataset = dataset
        self.mod = model
        self.opt = optimizer
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_accuracy = []
        self.validation_accuracy = []
        self.train_loss = []
        self.validation_loss = []

  def back_prop(self, label, device):
    print("Device:",device)
    loss_per_epoch = 0
    accuracy_per_epoch = 0
    self.model.train()
    for (x, y) in label:
        
        x = x.to(device)
        y = y.to(device)
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss_var = self.criterion(y_pred, y)
        loss_var.backward()
        accuracy_var = calc_accuracy(y_pred, y)
        self.optimizer.step()
        loss_per_epoch += loss_var.item()
        accuracy_per_epoch += accuracy_var.item()

    return loss_per_epoch / len(label), accuracy_per_epoch / len(label)


  def train_model(self,train_data_iterator,valid_data_iterator,device,optimizer,num_epochs,learning_rate,wd):
        
        if(self.dataset == 'CIFAR10'):
         num_output = 10
        elif (self.dataset == 'FDDB'):
         num_output = 2
        
        if(self.mod == 'AlexNet'):
         model = AlexNet(num_output)
          #print(model)
        elif self.model == 'VGG': 
         model = VGG_net(num_output)
    
        if self.opt == 'SGD':
         optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd, momentum = 0.9)
        elif self.opt == 'Adam':
         optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

        criterion = nn.CrossEntropyLoss()

        model = model.to(device)
        criterion = criterion.to(device)
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        ideal_loss = float('inf')
        
        for epoch in tqdm(range(num_epochs)):
            
            start_time = time.time()
            
            training_loss, training_accuracy = self.back_prop(train_data_iterator, device)
            validation_loss, validation_accuracy = evaluate(model,valid_data_iterator, criterion, device)
            
            if validation_loss < ideal_loss:
                ideal_loss = validation_loss
                torch.save(model.state_dict(), 'alexnet_model.pt')
            
            end_time = time.time()

            time_elapsed = end_time - start_time
            epoch_mins = int(time_elapsed / 60)
            epoch_secs = int(time_elapsed - (epoch_mins * 60))

            self.train_accuracy.append(training_accuracy)
            self.train_loss.append(training_loss)
            self.validation_accuracy.append(validation_accuracy)
            self.validation_loss.append(validation_loss)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Training Loss: {training_loss:.3f} | Validation Loss: {validation_loss:.3f} | Learning rate: {learning_rate}')
            print(f'\t Training Accuracy: {training_accuracy*100:.2f}% |  Validation Accuracy: {validation_accuracy*100:.2f}%')
        if num_epochs != 1:
            plot_metrics(self.train_accuracy,self.validation_accuracy,self.train_loss,self.validation_loss,learning_rate)    

