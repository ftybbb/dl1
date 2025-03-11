import torch
import torch.nn as nn
import torch.nn.functional as F

# Mish Activation Function
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2, activation='mish', dropout=True):
        super(BasicBlock, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Dropout layer
        if dropout:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = None

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        # Mish activation function
        if activation=='mish':
            self.activation = Mish()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2, activation='mish', dropout=True):
        super(BottleneckBlock, self).__init__()

        # First 1x1 conv - reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv - increase channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # Dropout layer
        if dropout:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = None

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        # Mish activation function
        if activation == 'mish':
            self.activation = Mish()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        if self.dropout:
            out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ModifiedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, initial_channels=64, activation='mish', dropout=True, dropout_rate=0.3):
        super(ModifiedResNet, self).__init__()
        self.in_channels = initial_channels

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        if activation=='mish':
            self.activation = Mish()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        # Residual blocks
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1, dropout_rate=dropout_rate, dropout=dropout, activation=activation)
        self.layer2 = self._make_layer(block, initial_channels * 2, num_blocks[1], stride=2, dropout_rate=dropout_rate, dropout=dropout, activation=activation)
        self.layer3 = self._make_layer(block, initial_channels * 4, num_blocks[2], stride=2, dropout_rate=dropout_rate, dropout=dropout, activation=activation)

        # Output with global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(initial_channels * 4 * block.expansion, num_classes)
        if dropout:
            self.dropout = nn.Dropout(p=dropout_rate)  # Apply dropout before FC layer
        else:
            self.dropout = None

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate, dropout, activation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropout_rate, activation, dropout))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        if self.dropout is not None:
            out = self.dropout(out)  # Apply dropout before FC layer
        out = self.fc(out)
        return out

# CIFAR-10 ResNet variant with Mish activation & Dropout
def improved_resnet_cifar(num_classes=10, model_size='medium', activation='mish', dropout=True):
    """
    Create a ResNet model with Mish activation and Dropout for CIFAR-10.

    Args:
        num_classes: Number of output classes
        model_size: Size of model - 'small', 'medium', or 'large'

    Returns:
        ModifiedResNet model
    """
    if model_size == 'small':
        return ModifiedResNet(BasicBlock, [2, 2, 2], num_classes, initial_channels=32, dropout_rate=0.2, activation=activation, dropout=dropout)
    elif model_size == 'medium':
        return ModifiedResNet(BasicBlock, [3, 4, 3], num_classes, initial_channels=64, dropout_rate=0.3, activation=activation, dropout=dropout)
    elif model_size == 'large':
        return ModifiedResNet(BottleneckBlock, [3, 4, 3], num_classes, initial_channels=48, dropout_rate=0.4, activation=activation, dropout=dropout)
    else:
        raise ValueError(f"Unknown model size: {model_size}")