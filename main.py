import os, sys
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
import timm

from models import *
from train import *
from metrics import *
from dataset import *

def select_model(choice=1):
    if choice==1:
        return mobilenet(num_classes=2)
    elif choice==2:
        return efficientnet(num_classes=2)
    elif choice==3:
        return inceptionv3(num_classes=2)


train_dataloader,test_dataloader=create_loaders()
print(f'Training dataset size: {len(train_dataloader)}')
print(f'Test dataset size: {len(test_dataloader)}')
model= select_model(1)
# Create optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

mobilenet_results =train(model=model,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=test_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=3,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                              save_path='models/best_model_mobilenet.pth')
y_true = []
y_pred = []
y_scores=[]
# Initialize your model (efficientnet_b0_model) and test_loader

# Rest of your code to get predictions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
for inputs, labels in test_dataloader:
    output1 = model(inputs.to(device))  # Feed Network
    probs_1 = torch.softmax(output1, dim=1)
    _, predicted_class = torch.max(probs_1, 1)
    labels = labels.data.numpy()
    y_true.extend(labels)
    y_pred.extend(predicted_class.cpu().data.numpy())
    y_scores.extend(probs_1.cpu().data.numpy())
ops(y_true,y_pred,y_scores,save_folder='results',num_classes=2)