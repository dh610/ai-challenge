import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelBinarizer
from model import MyModel
import sys

repository_path = '/content/drive/MyDrive/ai-challenge/'

train_images = np.load(repository_path + 'data/trainset.npy')
train_labels = np.load(repository_path + 'data/trainlabel.npy')
test_images = np.load(repository_path + 'data/testset.npy')

# One-hot encoding
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)

train_images_tensor = torch.tensor(train_images).float()
train_labels_tensor = torch.tensor(train_labels).float()
test_images_tensor = torch.tensor(test_images).float()

train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

num_classes = len(lb.classes_)
model = MyModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

if not torch.cuda.is_available():
    print("CUDA is disabled")
    sys.exit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
model_save_path = repository_path + "weight/final_model_weights.pth"

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.permute(0, 3, 1, 2)
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, torch.max(labels, 1)[1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    train_loss =  running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}.")

model.eval()
test_loader = DataLoader(test_images_tensor, batch_size=16, shuffle=False)
test_predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_predictions.append(predicted.cpu().numpy())

test_predictions = np.concatenate(test_predictions)
np.save(repository_path + 'test_predictions.npy', test_predictions)
