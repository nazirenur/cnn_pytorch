
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from PIL import Image

from data.data_loader import load_data
from models.model import CNNModel
from torch.utils.data import DataLoader
from utils.visualization import plot_sample_image, plot_loss_accuracy
from models.train import train_model

test = load_data('fashion-mnist_test.csv')
train = load_data('fashion-mnist_train.csv')



targets_numpy = train.label.values
features_numpy = train.loc[:, train.columns != "label"].values / 255  # normalization

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                              targets_numpy,                                                                              test_size=0.2,
                                                                              random_state = 42)

# Convert to PyTorch tensors
featuresTrain = torch.from_numpy(features_train).float()
targetsTrain = torch.from_numpy(targets_train).long()  # LongTensor for class labels

featuresTest = torch.from_numpy(features_test).float()
targetsTest = torch.from_numpy(targets_test).long()  # LongTensor for class labels


# Create DataLoader
batch_size = 100
train_data = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test_data = torch.utils.data.TensorDataset(featuresTest, targetsTest)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Visualize a sample image
plot_sample_image(features_numpy, targets_numpy, index=1)  # You can change the index as needed

# Training settings
num_epochs = 2500 // (len(features_train) // batch_size)
learning_rate = 0.1

# Train the model
model, loss_list, iteration_list, accuracy_list = train_model(train_loader, test_loader, num_epochs, learning_rate)


plot_loss_accuracy(iteration_list, loss_list, accuracy_list)
