import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import MyModel
import sys

repository_path = '/content/drive/MyDrive/ai-challenge/'
train_labels = np.load(repository_path + 'data/trainlabel.npy')

test_images = np.load(repository_path + 'data/testset.npy')
test_images_tensor = torch.tensor(test_images).float()

model_save_path = repository_path + "weight/final_model_weights.pth"

num_classes = len(np.unique(train_labels))
model = MyModel(num_classes)
model.load_state_dict(torch.load(model_save_path))

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
csv_save_path = repository_path + 'test_predictions.csv'
predictions_df.to_csv(csv_save_path)
print(f"Test predictions saved to '{csv_save_path}'.")