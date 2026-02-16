
import torch.nn as nn
import torch


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from robustx.lib.models.BaseModel import BaseModel
import torch
import os


# Define the neural network model
class Classifier(nn.Module):
    def __init__(self, X_train_shape):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(X_train_shape, round(X_train_shape / 2))
        self.fc2 = nn.Linear(round(X_train_shape / 2), 2)  # Number of classes in target

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, X_train_shape):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(X_train_shape, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x



