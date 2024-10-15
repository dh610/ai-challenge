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

class MyModel(nn.Module):
    def __init__(self, block, num_classes=100):
        super(MyModel, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Sequential(
            DSC(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
        )

        self.conv2_x = self._make_layer(block, 32, 2, 1)
        self.conv3_x = self._make_layer(block, 64, 2, 2)
        self.conv4_x = self._make_layer(block, 128, 1, 2)
        self.conv5_x = self._make_layer(block, num_classes, 1, 2)

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
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)

class MyNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MyNetBlock, self).__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        # First branch: Depthwise + Pointwise convolution
        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Identity()  # If stride = 1, the first branch remains unchanged

        # Second branch: Pointwise + Depthwise convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride == 2 else mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // 2

        # Split the channels into two groups and shuffle them
        x = x.view(batch_size, 2, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)  # Concatenate the two branches
        else:
            x1, x2 = x.chunk(2, dim=1)  # Split channels into two parts
            out = torch.cat((x1, self.branch2(x2)), 1)

        return self.channel_shuffle(out)  # Perform channel shuffle

class MyNet(nn.Module):
    def __init__(self, num_classes=100):
        super(MyNet, self).__init__()
        # Output channels for different model sizes
        out_channels = (16, 32, 64, 128, num_classes)

        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # Max pooling layer

        # Stacked ShuffleNetV2 blocks for different stages
        self.stage2 = self._make_stage(out_channels[0], out_channels[1], 4)
        self.stage3 = self._make_stage(out_channels[1], out_channels[2], 12)
        self.stage4 = self._make_stage(out_channels[2], out_channels[3], 5)

        # Final 1x1 convolution layer before classification
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[3], out_channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels[4]),
            nn.ReLU(inplace=True)
        )

    def _make_stage(self, in_channels, out_channels, num_blocks):
        # The first block in each stage has stride=2, the rest have stride=1
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(MyNetBlock(in_channels, out_channels, stride))
            in_channels = out_channels  # Update in_channels for the next block
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)  # Initial convolution
        x = self.maxpool(x)  # Max pooling
        x = self.stage2(x)  # Stage 2
        x = self.stage3(x)  # Stage 3
        x = self.stage4(x)  # Stage 4
        x = self.conv5(x)  # Final 1x1 convolution
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # Adaptive average pooling
        return x
