import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from torchsummary import summary

from utils import load_latest_ckpt, TestDataset
from model import MyModel, BasicBlock, MyNet

train_labels = np.load('data/trainlabel.npy')
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
model = MyModel(BasicBlock).to('cuda')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")
summary(model, (3, 32, 32))
sys.exit(0)
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