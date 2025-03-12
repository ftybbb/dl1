import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish Activation Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# SE Block for channel attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2, activation='swish', 
                 use_se=True, se_reduction=16, dropout=True):
        super(BasicBlock, self).__init__()
        
        # Choose activation function
        if activation == 'swish':
            self.activation = Swish()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SiLU(inplace=True)  # SiLU is PyTorch's Swish

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE attention block
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels, reduction=se_reduction)
        
        # Dropout layer
        self.use_dropout = dropout
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        if self.use_dropout:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)
            
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2, activation='swish', 
                 use_se=True, se_reduction=16, dropout=True):
        super(BottleneckBlock, self).__init__()
        
        # Choose activation function
        if activation == 'swish':
            self.activation = Swish()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SiLU(inplace=True)  # SiLU is PyTorch's Swish
            
        # First 1x1 conv - reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv - increase channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # SE attention block
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels * self.expansion, reduction=se_reduction)
        
        # Dropout layer
        self.use_dropout = dropout
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_se:
            out = self.se(out)
            
        if self.use_dropout:
            out = self.dropout(out)
            
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out

class ImprovedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, initial_channels=64, activation='swish',
                 dropout=True, dropout_rate=0.3, use_se=True, se_reduction=16, stem_channels=32):
        super(ImprovedResNet, self).__init__()
        self.in_channels = initial_channels
        
        # Choose activation function
        if activation == 'swish':
            self.activation = Swish()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SiLU(inplace=True)  # SiLU is PyTorch's Swish

        # Enhanced stem - improved initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            self.activation,
            nn.Conv2d(stem_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            self.activation
        )
        
        # Residual blocks with progressive feature maps
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1, 
                                       dropout_rate=dropout_rate, activation=activation,
                                       use_se=use_se, se_reduction=se_reduction, dropout=dropout)
        self.layer2 = self._make_layer(block, initial_channels * 2, num_blocks[1], stride=2, 
                                       dropout_rate=dropout_rate, activation=activation,
                                       use_se=use_se, se_reduction=se_reduction, dropout=dropout)
        self.layer3 = self._make_layer(block, initial_channels * 4, num_blocks[2], stride=2, 
                                       dropout_rate=dropout_rate, activation=activation,
                                       use_se=use_se, se_reduction=se_reduction, dropout=dropout)
        self.layer4 = self._make_layer(block, initial_channels * 8, num_blocks[3], stride=2, 
                                      dropout_rate=dropout_rate, activation=activation,
                                      use_se=use_se, se_reduction=se_reduction, dropout=dropout)
        
        # Global average pooling and classifier with dropout
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature aggregation
        self.classifier = nn.Sequential(
            nn.Linear(initial_channels * 8 * block.expansion, 1024),
            self.activation,
            nn.Dropout(dropout_rate) if dropout else nn.Identity(),
            nn.Linear(1024, num_classes)
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate, activation, 
                    use_se, se_reduction, dropout):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropout_rate, activation, 
                               use_se, se_reduction, dropout))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out

# Efficient version for CIFAR-10 with under 5M parameters
def improved_resnet_cifar(num_classes=10, model_size='medium', activation='swish', dropout=True, dropout_rate=0.3):
    """
    Create an improved ResNet model for CIFAR-10 with better generalization.
    
    Args:
        num_classes: Number of output classes
        model_size: Size of model - 'small', 'medium', or 'large'
        activation: 'swish' or 'relu'
        dropout: Whether to use dropout
        dropout_rate: Dropout rate if dropout is used
        
    Returns:
        ImprovedResNet model
    """
    if model_size == 'small':
        # Small model with BasicBlock - ~2.4M parameters
        return ImprovedResNet(
            BasicBlock, [2, 2, 2, 2], 
            num_classes=num_classes, 
            initial_channels=32, 
            activation=activation, 
            dropout=dropout, 
            dropout_rate=0.2,
            use_se=True,
            se_reduction=8,
            stem_channels=16
        )
    elif model_size == 'medium':
        # Medium model with BasicBlock - ~4.2M parameters
        return ImprovedResNet(
            BasicBlock, [3, 4, 6, 3], 
            num_classes=num_classes, 
            initial_channels=48, 
            activation=activation, 
            dropout=dropout, 
            dropout_rate=0.3,
            use_se=True,
            se_reduction=16,
            stem_channels=24
        )
    elif model_size == 'large':
        # Larger model with BottleneckBlock - ~4.9M parameters
        return ImprovedResNet(
            BottleneckBlock, [3, 4, 6, 3], 
            num_classes=num_classes, 
            initial_channels=40, 
            activation=activation, 
            dropout=dropout, 
            dropout_rate=0.4,
            use_se=True,
            se_reduction=16,
            stem_channels=32
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")
        
# Training recommendations:
"""
To achieve >90% test accuracy on CIFAR-10 with this model, consider:

1. Data Augmentation:
   - RandomCrop(32, padding=4)
   - RandomHorizontalFlip()
   - ColorJitter(brightness=0.2, contrast=0.2)
   - RandomRotation(15)
   - Use CutMix or MixUp augmentation

2. Training strategy:
   - Use Cosine Annealing LR schedule with warm restarts
   - Initial LR: 0.1 with SGD (momentum=0.9, weight_decay=5e-4)
   - Batch size: 128
   - Train for 200 epochs
   - Use Label Smoothing (0.1)

3. Regularization:
   - Weight decay: 5e-4
   - Stochastic Depth: 0.2 drop probability
   - Gradient clipping: 1.0

Example training loop pseudocode:
```python
# Setup model
model = improved_resnet_cifar(model_size='medium', activation='swish')
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Train for 200 epochs
for epoch in range(200):
    train_one_epoch(model, train_loader, optimizer, criterion)
    val_acc = validate(model, val_loader)
    scheduler.step()
    
    # Apply early stopping if needed
```
"""