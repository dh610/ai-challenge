import torch.nn as nn
import torch
import torch.nn.functional as F

# Depthwise Separable Convolution
class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        super(DSC, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.residual_function = nn.Sequential(
            DSC(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DSC(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                DSC(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        return self.relu(x)

class NeoResNet(nn.Module):
    def __init__(self, block, num_classes=100):
        super(NeoResNet, self).__init__()
        self.in_channels = 3

        '''
        self.conv1 = nn.Sequential(
            #DSC(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
        )
        '''

        self.conv1 = self._make_layer(block, 4, 1)
        self.conv1_x = self._make_layer(block, 8, 3)
        self.conv2_x = self._make_layer(block, 16, 3)
        self.conv3_x = self._make_layer(block, 32, 3)
        self.conv4_x = self._make_layer(block, 64, 3)
        self.conv5_x = self._make_layer(block, 128, 1, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def MyModel():
    return NeoResNet(BasicBlock)
