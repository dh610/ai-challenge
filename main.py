import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from model import MyModel
from utils import load_latest_ckpt
import sys

from train import train, load_and_concat, evaluate


def main():
    # Load datasets
    train_loader, val_loader, num_classes = load_and_concat()

    if not torch.cuda.is_available():
        print("CUDA is disabled")
        sys.exit(1)
    device = torch.device("cuda")

    model = MyModel(num_classes).to(device)
    model, start_epoch = load_latest_ckpt(model, "weight/")
    #start_epoch = 0
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    scheduler = optim.CosineAnnealingLR(optimizer, T_max=20)

    num_epochs = 1000
    model_save_path = "weight/epoch_"

    prev_acc = 0
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train(device, model, criterion, optimizer, train_loader, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], "                                           \
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "       \
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if prev_acc <= val_acc:
            prev_acc = val_acc
            if prev_acc < 0.5:
                continue

            tmp_save_path = model_save_path + f"{epoch+1}.pth"
            torch.save(model.state_dict(), tmp_save_path)
            print(f"Model weights saved to {tmp_save_path}.")

    return

if __name__ == '__main__':
    random.seed(10)
    np.random.seed(123)
    main()
