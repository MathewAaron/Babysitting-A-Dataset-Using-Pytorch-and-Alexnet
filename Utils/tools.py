# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:39:41 2022

@author: Dell
"""
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
    
#functions used to evaluate loss, accuracy and plot graphs.

#Function to calculate accuracy
def calc_accuracy(y_prediction, y):
    top_prediction = y_prediction.argmax(1, keepdim = True)
    correct = top_prediction.eq(y.view_as(top_prediction)).sum()
    accuracy = correct.float() / y.shape[0]
    return accuracy

#Function to evaluate loss and accuracy
def evaluate(model, data_loader, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in data_loader:

            x = x.to(device)
            y = y.to(device)

            y_prediction = model(x)

            loss = criterion(y_prediction, y)

            acc = calc_accuracy(y_prediction, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)
  

#Function to plot training and valid set accuracy and loss for given learning rate. 
def plot_metrics(Train_Acc,Val_Acc,Tain_Loss,Val_Loss,learning_rate):
      print('plot')
      plt.plot(np.asarray(Train_Acc),'r',label="Training Accuracy")
      plt.plot(np.asarray(Val_Acc),'b',label="Validation Accuracy")
      plt.title("Training and Validation Accuracy, LR: {}".format(learning_rate))
      plt.xlabel("No. Epochs")
      plt.ylabel("Accuracy")
      plt.legend(loc='best')
      plt.show()
    
      plt.plot(np.asarray(Tain_Loss),'r',label="Training Loss")
      plt.plot(np.asarray(Val_Loss),'b',label="Validation Loss")
      plt.title("Training and Validation Loss, LR: {}".format(learning_rate))
      plt.xlabel("No. Epochs")
      plt.ylabel("Loss")
      plt.legend(loc='best')
      plt.show()
