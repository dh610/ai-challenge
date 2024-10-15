import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import random

from model import MyModel, BasicBlock
from utils import AugmentedDataset
from augmentation import cutmix_data, mixup_data
import sys

# 데이터 로드
train_images = np.load('data/trainset.npy')
train_labels = np.load('data/trainlabel.npy')
test_images = np.load('data/testset.npy')

mean = [129.4377 / 255.0, 124.1342 / 255.0, 112.4572 / 255.0]  # [0, 1]
std = [68.2042 / 255.0, 65.4584 / 255.0, 70.4745 / 255.0]

# 데이터 증강 및 정규화
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop(32, padding=2),
    transforms.ToTensor(),  # [0, 255] → [0, 1]
    transforms.Normalize(mean=mean, std=std),
])

# AugmentedDataset 클래스 정의
train_dataset = AugmentedDataset(train_images, train_labels, transform=augmentation)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

num_classes = len(np.unique(train_labels))
model = MyModel(BasicBlock, [3, 1, 2], num_classes)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

if not torch.cuda.is_available():
    print("CUDA is disabled")
    sys.exit(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 100
model_save_path = "weight/epoch_"

model.train()
early_stopping_cnt = 0

for epoch in range(num_epochs):
    prev_loss = float('inf')
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).long()

        if epoch >= 300:
            if random.random() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(device, images, labels)
            else:
                images, targets_a, targets_b, lam = cutmix_data(device, images, labels)

            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:

            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

    scheduler.step()
    train_loss =  running_loss / len(train_loader)
    if train_loss < prev_loss:
        early_stopping_cnt += 1
    prev_loss = train_loss
    train_accuracy = running_correct / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    if (epoch+1) % 10 == 0:
        tmp_save_path = model_save_path + f"{epoch+1}.pth"
        torch.save(model.state_dict(), tmp_save_path)
        print(f"Model weights saved to {tmp_save_path}.")


