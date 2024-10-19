import os
import torch
from torch.utils.data import TensorDataset
from PIL import Image
import numpy as np

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
        image = Image.fromarray(image.astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        return image, label

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

def warmup_scheduler(optimizer, warmup_epochs, total_epochs, warmup_factor=1e-3):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # warmup step
            return warmup_factor + (1 - warmup_factor) * (current_epoch / warmup_epochs)
        else:
            # cosine annealing step
            return 0.5 * (1 + np.cos(np.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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