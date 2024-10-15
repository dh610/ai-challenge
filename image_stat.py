import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

train_images = np.load('data/trainset.npy')
train_labels = np.load('data/trainlabel.npy')
test_images = np.load('data/testset.npy')

train_images_tensor = torch.tensor(train_images).float()
train_labels_tensor = torch.tensor(train_labels).float()
test_images_tensor = torch.tensor(test_images).float()

print(f"Train images shape: {train_images_tensor.shape}")

train_images_flat = train_images_tensor.permute(3, 0, 1, 2).reshape(3, -1)

mean = train_images_flat.mean(dim=1)
std = train_images_flat.std(dim=1)

print(f"Mean: {mean}, Std: {std}")