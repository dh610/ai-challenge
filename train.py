import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import random
from sklearn.model_selection import train_test_split

from model import MyModel
from utils import AugmentedDataset
from augmentation import cutmix_data, mixup_data
from utils import load_latest_ckpt, warmup_scheduler
from loss import SupConLoss
import sys

random.seed(10)
np.random.seed(123)

# 데이터 로드
train_images = np.load('data/trainset.npy')
train_labels = np.load('data/trainlabel.npy')
test_images = np.load('data/testset.npy')

mean = [129.4377 / 255.0, 124.1342 / 255.0, 112.4572 / 255.0]  # [0, 1]
std = [68.2042 / 255.0, 65.4584 / 255.0, 70.4745 / 255.0]

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),  # [0, 255] → [0, 1]
    transforms.Normalize(mean=mean, std=std),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] → [0, 1]
    transforms.Normalize(mean=mean, std=std),
])

train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42, stratify=train_labels
)

# AugmentedDataset 클래스 정의
train_dataset = AugmentedDataset(train_images, train_labels, transform=augmentation)
val_dataset = AugmentedDataset(val_images, val_labels, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

num_classes = len(np.unique(train_labels))
model = MyModel()
model, start_epoch = load_latest_ckpt(model, "weight/")
#start_epoch = 0
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")

criterion = nn.CrossEntropyLoss()
# criterion = SupConLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = warmup_scheduler(10, 100)

if not torch.cuda.is_available():
    print("CUDA is disabled")
    sys.exit(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 1000
model_save_path = "weight/epoch_"

# Functino to evlauate
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

prev_acc = 0
for epoch in range(start_epoch, num_epochs):
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

    # scheduler.step()
    train_loss =  running_loss / len(train_loader)
    train_accuracy = running_correct / total_samples
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], "                                           \
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "       \
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    if prev_acc <= val_acc:
        prev_acc = val_acc
        if prev_acc < 0.5:
            continue

        tmp_save_path = model_save_path + f"{epoch+1}.pth"
        torch.save(model.state_dict(), tmp_save_path)
        print(f"Model weights saved to {tmp_save_path}.")


