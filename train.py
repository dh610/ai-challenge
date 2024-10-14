import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import MyModel, BasicBlock
import sys

repository_path = '/content/drive/MyDrive/ai-challenge/'

train_images = np.load(repository_path + 'data/trainset.npy')
train_labels = np.load(repository_path + 'data/trainlabel.npy')
test_images = np.load(repository_path + 'data/testset.npy')

train_images_tensor = torch.tensor(train_images).float()
train_labels_tensor = torch.tensor(train_labels).float()
test_images_tensor = torch.tensor(test_images).float()

train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

num_classes = len(np.unique(train_labels))
model = MyModel(BasicBlock, [2, 2, 1, 1], num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

if not torch.cuda.is_available():
    print("CUDA is disabled")
    sys.exit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 1000
model_save_path = repository_path + "weight/model_weights"

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    for images, labels in tqdm(train_loader):
        images = images.permute(0, 3, 1, 2)
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

    if (epoch+1) % 50 == 0:
        tmp_save_path = model_save_path + f"{epoch+1}.pth"
        torch.save(model.state_dict(), tmp_save_path)
        print(f"Model weights saved to {tmp_save_path}.")

