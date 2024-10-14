import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from model import MyModel, BasicBlock
import sys

# 데이터 로드
train_images = np.load('data/trainset.npy')
train_labels = np.load('data/trainlabel.npy')
test_images = np.load('data/testset.npy')

train_images_tensor = torch.tensor(train_images).permute(0, 3, 1, 2).float()  # (N, C, H, W)
train_labels_tensor = torch.tensor(train_labels).long()

train_images_tensor /= 255.0

mean = [129.4377 / 255.0, 124.1342 / 255.0, 112.4572 / 255.0]  # [0, 1]
std = [68.2042 / 255.0, 65.4584 / 255.0, 70.4745 / 255.0]

# 데이터 증강 및 정규화
augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=mean, std=std)  # 정규화
])

# AugmentedDataset 클래스 정의
class AugmentedDataset(TensorDataset):
    def __init__(self, images, labels, transform=None):
        super(AugmentedDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = AugmentedDataset(train_images_tensor, train_labels_tensor, transform=augmentation)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

num_classes = len(np.unique(train_labels))
model = MyModel(BasicBlock, [2, 2, 1, 1], num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

if not torch.cuda.is_available():
    print("CUDA is disabled")
    sys.exit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 100
model_save_path = "weight/epoch_"

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).long()

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
    train_accuracy = running_correct / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    if (epoch+1) % 10 == 0:
        tmp_save_path = model_save_path + f"{epoch+1}.pth"
        torch.save(model.state_dict(), tmp_save_path)
        print(f"Model weights saved to {tmp_save_path}.")

