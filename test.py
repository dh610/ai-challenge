import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import MyModel
import sys
import os

def load_latest_ckpt(net, ckpt_path):
    checkpoint_files = [f for f in os.listdir(ckpt_path) if f.startswith('model_weights') and f.endswith('.pth')]

    if not checkpoint_files:
        print("No checkpoint found.")
        return net, 0

    epoch_values = [int(f.split('_')[1].split('.')[0]) for f in checkpoint_files]

    latest_epoch = max(epoch_values)
    latest_checkpoint = f'model_weights{latest_epoch}.pth'

    print(f"Loading checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(os.path.join(ckpt_path, latest_checkpoint))
    print(checkpoint.keys())

    checkpoint = torch.load(os.path.join(ckpt_path, latest_checkpoint))
    net.load_state_dict(checkpoint)

    return net, latest_epoch 

train_labels = np.load('data/trainlabel.npy')

test_images = np.load('data/testset.npy')
test_images_tensor = torch.tensor(test_images).float()

model_save_path = "weight/"

num_classes = len(np.unique(train_labels))
model = MyModel(num_classes)
model, _ = load_latest_ckpt(model, model_save_path)

if not torch.cuda.is_available():
    print("CUDA is disabled")
    sys.exit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
test_loader = DataLoader(test_images_tensor, batch_size=16, shuffle=False)
test_predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.permute(0, 3, 1, 2)
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