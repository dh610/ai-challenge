import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from torchsummary import summary

from utils import load_latest_ckpt, TestDataset
from model import MyModel

train_labels = np.load('data/trainset.npy')
test_images = np.load('data/testset.npy')

dummy_labels = np.zeros(len(test_images))

test_transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] → [0, 1]
    transforms.Normalize(mean=[129.4377 / 255.0, 124.1342 / 255.0, 112.4572 / 255.0],
                         std=[68.2042 / 255.0, 65.4584 / 255.0, 70.4745 / 255.0])
])

test_dataset = TestDataset(test_images, dummy_labels, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model_save_path = "weight/"

num_classes = len(np.unique(train_labels))

if not torch.cuda.is_available():
    print("CUDA is disabled")
    sys.exit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")
model, _ = load_latest_ckpt(model, model_save_path)
summary(model, input_size=(3, 32, 32))


model.eval()
test_predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_predictions.append(predicted.cpu().numpy())

test_predictions = np.concatenate(test_predictions)
predictions_df = pd.DataFrame(test_predictions, columns=["label"])
predictions_df.index.name = 'id_idx'
csv_save_path = 'result/test_predictions.csv'
predictions_df.to_csv(csv_save_path)
print(f"Test predictions saved to '{csv_save_path}'.")