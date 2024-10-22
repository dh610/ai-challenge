import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import random

from utils import AugmentedDataset
from augmentation import cutmix_data, mixup_data
import sys

def load_and_concat():
    # Load datesets
    train_images = np.load('data/testset.npy')
    train_labels = np.load('data/testlabel.npy')
    val_images = np.load('data/testset.npy')
    val_labels = np.load('data/testlabel.npy')

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),  # [0, 255] → [0, 1]
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] → [0, 1]
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = AugmentedDataset(train_images, train_labels, transform=augmentation)
    val_dataset = AugmentedDataset(val_images, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    return train_loader, val_loader, len(np.unique(train_labels))

def train(device, model, criterion, optimizer, train_loader, scheduler = 'None'):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).long()
        '''
        outputs = model(images)
        loss = criterion(outputs, labels)

        '''
        if random.random() < 0.7:
            images, labels_a, labels_b, lam = cutmix_data(device, images, labels, alpha=1.0)
        else:
            images, labels_a, labels_b, lam = mixup_data(device, images, labels, alpha=1.0)
        outputs = model(images)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        #'''

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

    if scheduler != 'None':
        scheduler.step()
    train_loss =  running_loss / len(train_loader)
    train_accuracy = running_correct / total_samples
    return train_loss, train_accuracy

# Functinon to evlauate
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_accuracy = running_correct / total_samples
    return val_loss, val_accuracy
