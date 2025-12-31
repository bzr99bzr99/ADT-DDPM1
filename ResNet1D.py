import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion = 1

    # Add dropout_p parameter to __init__
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # Add a dropout layer - placed after the second BN, before the shortcut addition
        self.dropout = nn.Dropout(dropout_p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply dropout after the second batch norm
        out = self.dropout(out)

        out += self.shortcut(identity)
        out = self.relu(out) # Final ReLU activation

        return out

class ResNet1D(nn.Module):
    # Add dropout_p parameter to __init__
    def __init__(self, block, layers, num_classes=1, dropout_p=0.0): # Default dropout_p=0.0 (no dropout)
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        # Store dropout_p to pass it to _make_layer
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Pass dropout_p when creating layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d((1))
        # Add a dropout layer before the final fully connected layer
        self.fc_dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # _make_layer now needs to accept and pass dropout_p
    def _make_layer(self, block, out_channels, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        # Pass dropout_p to the block constructor
        layers.append(block(self.in_channels, out_channels, stride, dropout_p=self.dropout_p))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
             # Pass dropout_p to the block constructor for subsequent blocks
            layers.append(block(self.in_channels, out_channels, stride=1, dropout_p=self.dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Apply dropout before the final fully connected layer
        x = self.fc_dropout(x)
        x = self.fc(x)

        return x

# Modify the factory function to accept dropout_p
def resnet1d_18(num_classes=1, dropout_p=0.0):
    """Constructs a ResNet-18 model for 1D data with optional dropout.

    Args:
        num_classes (int): Number of output classes (e.g., 1 for regression). Default is 1.
        dropout_p (float): Dropout probability. Applied within blocks and before the FC layer.
                           Default is 0.0 (no dropout).
    """
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes, dropout_p=dropout_p)

