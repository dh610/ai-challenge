import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        print(f"Before conv1: {x.shape}")
        x = self.pool(self.relu(self.conv1(x)))
        print(f"After conv1: {x.shape}")
        x = self.pool(self.relu(self.conv2(x)))
        print(f"After conv2: {x.shape}")
        x = self.pool(self.relu(self.conv3(x)))
        print(f"After conv3: {x.shape}")
        x = x.reshape(-1, 32 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x