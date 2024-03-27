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

train_dataloader,val_dataloader,test_dataloader=create_loaders('/content/drive/MyDrive/fusedmammonet/ddsm')
print(f'Training dataset size: {len(train_dataloader)}')
print(f'Test dataset size: {len(test_dataloader)}')


select_model={1:mobilenet(num_classes=5),
              2:efficientnet(num_classes=5),
              3:inceptionv3(num_classes=5),
              4:ensemble(mobilenet_tl(num_classes=5),efficientnet_tl(num_classes=5),inceptionv3_tl(num_classes=5),num_classes=5),
              5:ensemble_tl(mobilenet_tl(num_classes=5),efficientnet_tl(num_classes=5),inceptionv3_tl(num_classes=5),num_classes=5)
              }

model= select_model[1]
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

results =train(model=model,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=val_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=10,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                              save_path='/content/drive/MyDrive/fusedmammonet/models/best_model_mobilenet.pth')
plot_and_save_loss_curves(results,save_folder='/content/drive/MyDrive/fusedmammonet/results', save_name="mobilenet_loss_curves")

model= select_model[2]
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

results =train(model=model,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=val_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=10,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                              save_path='/content/drive/MyDrive/fusedmammonet/models/best_model_efficientnet.pth')
plot_and_save_loss_curves(results,save_folder='/content/drive/MyDrive/fusedmammonet/results', save_name="efficientnet_loss_curves")

model= select_model[3]
# Create optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

results =train(model=model,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=val_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=10,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                              save_path='/content/drive/MyDrive/fusedmammonet/models/best_model_inceptionv3.pth')
plot_and_save_loss_curves(results,save_folder='/content/drive/MyDrive/fusedmammonet/results', save_name="inceptionv3_loss_curves")

model= select_model[4]
# Create optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

results =train(model=model,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=val_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=10,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                              save_path='/content/drive/MyDrive/fusedmammonet/models/best_model_ensemble.pth')
plot_and_save_loss_curves(results,save_folder='/content/drive/MyDrive/fusedmammonet/results', save_name="loss_curves")

model= select_model[5]
y_true = []
y_pred = []
y_scores=[]
# Initialize your model (efficientnet_b0_model) and test_loader

# Rest of your code to get predictions
#model=model.to('cpu')
for inputs, labels in test_dataloader:
    output1 = model(inputs.to('cpu'))  # Feed Network
    probs_1 = torch.softmax(output1, dim=1)
    _, predicted_class = torch.max(probs_1, 1)
    labels = labels.data.numpy()
    y_true.extend(labels)
    y_pred.extend(predicted_class.cpu().data.numpy())
    y_scores.extend(probs_1.cpu().data.numpy())
ops(y_true,y_pred,y_scores,save_folder='/content/drive/MyDrive/fusedmammonet/results',num_classes=5)
     
