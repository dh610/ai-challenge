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
import os

def load_latest_ckpt(net, ckpt_path):
    checkpoint_files = [f for f in os.listdir(ckpt_path) if f.startswith('epoch_') and f.endswith('.pth')]

    if not checkpoint_files:
        print("No checkpoint found.")
        return net, 0

    epoch_values = [int(f.split('_')[1].split('.')[0]) for f in checkpoint_files]

    latest_epoch = max(epoch_values)
    latest_checkpoint = f'epoch_{latest_epoch}.pth'

    print(f"Loading checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(os.path.join(ckpt_path, latest_checkpoint))
    print(checkpoint.keys())

    checkpoint = torch.load(os.path.join(ckpt_path, latest_checkpoint))
    net.load_state_dict(checkpoint)

    return net, latest_epoch 

train_labels = np.load('data/trainlabel.npy')
test_images = np.load('data/testset.npy')

dummy_labels = np.zeros(len(test_images))

test_transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] → [0, 1]
    transforms.Normalize(mean=[129.4377 / 255.0, 124.1342 / 255.0, 112.4572 / 255.0],
                         std=[68.2042 / 255.0, 65.4584 / 255.0, 70.4745 / 255.0])
])

class TestDataset(TensorDataset):
    def __init__(self, images, labels, transform=None):
        super(TestDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        # Transform 적용
        if self.transform:
            image = self.transform(image)

        return image, label

test_dataset = TestDataset(test_images, dummy_labels, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model_save_path = "weight/"

num_classes = len(np.unique(train_labels))
model = MyModel(BasicBlock, [2, 2, 1, 1], num_classes)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")
model, _ = load_latest_ckpt(model, model_save_path)

if not torch.cuda.is_available():
    print("CUDA is disabled")
    sys.exit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
test_predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_predictions.append(predicted.cpu().numpy())

test_predictions = np.concatenate(test_predictions)
predictions_df = pd.DataFrame(test_predictions, columns=["label"])
predictions_df.index.name = 'id_idx'
csv_save_path = 'result/test_predictions.csv'
predictions_df.to_csv(csv_save_path)
print(f"Test predictions saved to '{csv_save_path}'.")