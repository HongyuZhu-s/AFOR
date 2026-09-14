"""
EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces.

Reference:
    Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ.
    "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces."
    J Neural Eng. 2018 Oct;15(5):056013.

This is a standalone PyTorch implementation adapted from the original TensorFlow code:
https://github.com/vlawhern/arl-eegmodels
"""

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Module):
    """
    Conv2d layer with max-value weight constraint (clamp).
    Used for the depthwise spatial convolution to regularize spatial filters.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, max_value=1.0, bias=False, groups=1):
        super(Conv2dWithConstraint, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=bias, groups=groups)
        self.max_value = max_value

    def forward(self, x):
        self.conv.weight.data.clamp_(max=self.max_value)
        return self.conv(x)


class EEGNet(nn.Module):
    """
    EEGNet model for EEG-based emotion recognition.

    Args:
        num_electrodes: Number of EEG channels (default 62 for SEED, 32 for DEAP)
        datapoints: Number of time points per sample
        num_classes: Number of output classes (default 2 for binary)
        F1: Number of temporal filters (default 8)
        D: Depth multiplier for spatial filters (default 2)
        dropout: Dropout rate (default 0.5)
    """
    def __init__(self, num_electrodes=62, datapoints=128, num_classes=2,
                 F1=8, D=2, dropout=0.5):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.dropout = dropout

        # Block 1: Temporal convolution + Depthwise spatial convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.F1,
                               kernel_size=(1, datapoints // 2), padding='same',
                               bias=False)
        self.BN1 = nn.BatchNorm2d(self.F1)

        # Depthwise spatial convolution with max-norm constraint
        self.depth_conv = Conv2dWithConstraint(
            in_channels=self.F1, out_channels=self.F1 * self.D,
            kernel_size=(num_electrodes, 1), bias=False, groups=self.F1)

        self.BN2 = nn.BatchNorm2d(self.D * self.F1)
        self.act1 = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=4)
        self.dropout1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        F2 = self.D * self.F1
        self.sep_conv1 = nn.Conv2d(in_channels=self.D * self.F1,
                                   out_channels=self.D * self.F1,
                                   kernel_size=(1, 16), padding='same',
                                   bias=False, groups=self.D * self.F1)
        self.sep_conv2 = nn.Conv2d(in_channels=self.D * self.F1,
                                   out_channels=F2,
                                   kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=8)
        self.dropout2 = nn.Dropout(dropout)

        # Classifier
        self.fc = nn.Linear(F2 * (datapoints // 32), num_classes)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.depth_conv.conv.weight)
        nn.init.kaiming_normal_(self.sep_conv1.weight)
        nn.init.kaiming_normal_(self.sep_conv2.weight)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x shape: (batch_size, channels, datapoints)
        # Reshape to (batch_size, 1, channels, datapoints)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # Block 1
        x = self.conv1(x)          # (B, F1, C, T//2)
        x = self.BN1(x)
        x = self.depth_conv(x)     # (B, F1*D, 1, T//2)
        x = self.BN2(x)
        x = self.act1(x)
        x = self.pool1(x)          # (B, F1*D, 1, T//8)
        x = self.dropout1(x)

        # Block 2
        x = self.sep_conv1(x)      # (B, F1*D, 1, T//8)
        x = self.sep_conv2(x)      # (B, F2, 1, T//8)
        x = self.BN3(x)
        x = self.act2(x)
        x = self.pool2(x)          # (B, F2, 1, T//32)
        x = self.dropout2(x)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
