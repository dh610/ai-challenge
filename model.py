import torch.nn as nn

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

    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super(BasicBlock, self).__init__()
        
        self.residual_function = nn.Sequential(
            DSC(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            DSC(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                DSC(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels),
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        return self.relu(x)

class MyModel(nn.Module):
    def __init__(self, block, num_block, num_classes=100, groups=8):
        super(MyModel, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Sequential(
            DSC(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(groups, 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
        )

        self.conv2_x = self._make_layer(block, 32, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 128, num_block[2], 2)

        self.conv5_x= DSC(128, num_classes, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)
