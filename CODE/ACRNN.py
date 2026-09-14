"""
ACRNN: EEG-Based Emotion Recognition via Channel-Wise Attention and Self Attention.

Reference:
    W. Tao et al., "EEG-Based Emotion Recognition via Channel-Wise Attention
    and Self Attention," IEEE Trans. Affective Computing, 2023.

Original TensorFlow: https://github.com/AstoncPou/ACRNN

Adapted from the LibEER PyTorch implementation.
Modified to accept (batch, channels, timepoints) input directly,
and to output logits (compatible with CrossEntropyLoss).
"""

import torch
import torch.nn as nn
import numpy as np


def square(x):
    return x * x


def cov(x):
    x_t = x.permute([0, 1, 3, 2])
    return torch.matmul(x_t, x)


def safe_log(x, eps=1e-6):
    return torch.log(torch.clamp(x, min=eps))


class Expression(nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)


class Conv2dNormWeight(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dNormWeight, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dNormWeight, self).forward(x)


class CNN(nn.Module):
    def __init__(self, ic, ih, iw, kh, kw, ks, ph, pw, ps, oc):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, (kh, kw), ks),
            nn.ELU(),
            nn.MaxPool2d((ph, pw), ps),
        )
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        # x: (B, H, C, W)  →  conv expects (B, C, H, W) so permute
        x = x.permute(0, 1, 3, 2)       # (B, H, W, C)
        c = self.conv(x)                 # (B, oc, H', W')
        cd = self.dropout(c)
        return cd


class ChannelWiseAttention(nn.Module):
    def __init__(self, H, W, C, reduce):
        super(ChannelWiseAttention, self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.r = reduce
        self.fc = nn.Sequential(
            nn.Linear(self.C, self.r),
            nn.Tanh(),
            nn.Linear(self.r, self.C),
        )
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        # x: (B, H, W, C)
        x1 = x.permute(0, 3, 1, 2)                    # (B, C, H, W)
        mean = nn.AvgPool2d((1, self.W))
        feature_map = mean(x1).permute(0, 2, 3, 1)     # (B, H, 1, C)

        feature_map_fc = self.fc(feature_map)           # (B, H, 1, C)
        v = self.softmax(feature_map_fc)                # (B, H, 1, C)

        # Broadcast v across the width dimension
        vr = v.expand(-1, -1, self.W, -1)               # (B, H, W, C)
        channel_wise_attention_fm = x * vr
        return v, channel_wise_attention_fm


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTMBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
        )

    def forward(self, x, hidden0=None):
        # x: (B, seq_len, input_size)
        x = x.reshape(-1, 1, self.input_size)
        q, (hidden, cell) = self.lstm(x)
        h = hidden[1].reshape(-1, 1, self.hidden_dim)
        c = cell[1].reshape(-1, 1, self.hidden_dim)
        return h, c


class Dense(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim,
                 activation=lambda x: x):
        super().__init__()
        self.W1 = nn.Parameter(torch.Tensor(
            np.random.normal(size=(input_dim1, hidden_dim))))
        self.W2 = nn.Parameter(torch.Tensor(
            np.random.normal(size=(input_dim2, hidden_dim))))
        self.b = nn.Parameter(torch.Tensor(np.zeros(hidden_dim)))
        self.activation = activation
        self.vector = nn.Linear(input_dim2, input_dim2)

    def forward(self, x):
        y = self.vector(x)
        return self.activation(
            torch.matmul(x, self.W1) + torch.matmul(y, self.W2) + self.b)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.dense = Dense(input_dim, input_dim, input_dim)
        self.self_attention = nn.Sequential(
            nn.ELU(),
            nn.Linear(input_dim, input_dim),
        )
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        y = self.dense(x)
        z = self.self_attention(y)
        p = z * x
        p = self.softmax(p)
        A = p * x
        A = A.reshape(-1, A.shape[-1])
        A = self.dropout(A)
        return A


# ==============================================================================
# Main ACRNN model — adapted for (B, C, T) input
# ==============================================================================

class ACRNN(nn.Module):
    """
    ACRNN emotion recognition model.

    Args:
        n_channels:   number of EEG channels (e.g., 32 for DEAP, 62 for SEED)
        n_timepoints: number of time points per sample (e.g., 128 for DEAP)
        num_classes:  number of output classes (2 for DEAP binary, 3 for SEED)
    """

    def __init__(self, n_channels, n_timepoints, num_classes):
        super(ACRNN, self).__init__()

        # Internal dimensions
        self.H = 32                       # fixed internal spatial height
        self.W = n_timepoints             # time = width
        self.C = n_channels               # EEG channels
        self.reduce = 15

        # Sub-modules
        self.channel_wise_attention = ChannelWiseAttention(
            self.H, self.W, self.C, self.reduce)

        self.output_channel = 40
        self.kernel_height = n_channels
        self.kernel_width = 3
        self.kernel_stride = 1
        self.pooling_height = 1
        self.pooling_width = 1
        self.pooling_stride = 1

        self.cnn = CNN(self.H, self.C, self.W,
                       self.kernel_height, self.kernel_width,
                       self.kernel_stride,
                       self.pooling_height, self.pooling_width,
                       self.pooling_stride,
                       self.output_channel)

        self.hidden_dim = 64
        c_width = int((((n_timepoints - self.kernel_width)
                        // self.kernel_stride + 1)
                       - self.pooling_width) // self.pooling_stride + 1)
        c_width = max(c_width, 1)

        self.lstm_block = LSTMBlock(self.output_channel * c_width,
                                    self.hidden_dim)
        self.self_attention = SelfAttention(self.hidden_dim, 512)
        self.dropout = nn.Dropout(0.5)

        # Classifier — outputs logits (no Softmax, for CrossEntropyLoss)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, C, T)  →  reshape to model's expected (B, H, W, C)
        # H=32 is a fixed internal dimension
        x = x.unsqueeze(1).expand(-1, self.H, -1, -1)   # (B, 32, C, T)
        x = x.permute(0, 1, 3, 2)                        # (B, 32, T, C)

        _, x_ca = self.channel_wise_attention(x)          # (B, 32, T, C)
        x_cn = self.cnn(x_ca)                             # (B, 40, H', W')
        x_rn, x_c = self.lstm_block(x_cn)                 # (B, 1, 64) each
        x_sa = self.dropout(self.self_attention(x_rn))    # (B, 64)
        x_out = self.classifier(x_sa)                     # (B, num_classes)
        return x_out
