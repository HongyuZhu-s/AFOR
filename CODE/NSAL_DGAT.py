"""
NSAL-DGAT: Node-wise Spatial Attention with Domain Adversarial Graph
Attention Network for EEG emotion recognition.

Reference:
    Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10976537
    Code:  https://github.com/YYingDL/NSAL-DGAT

Adapted from the LibEER PyTorch implementation.
Uses Encoder + Classifier only (domain-adversarial components removed).
Accepts (batch, channels, timepoints) input and outputs logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.se(self.maxpool(x))
        avg_out = self.se(self.avgpool(x))
        return self.sigmoid(max_out + avg_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        return self.sigmoid(self.conv(result))


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class GATENet(nn.Module):
    def __init__(self, inc, reduction_ratio=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(inc, inc // reduction_ratio, bias=False),
            nn.ELU(inplace=False),
            nn.Linear(inc // reduction_ratio, inc, bias=False),
            nn.Tanh(),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.fc(x)


class resGCN(nn.Module):
    def __init__(self, inc, outc, band_num):
        super().__init__()
        self.GConv1 = nn.Conv2d(inc, outc, (1, 3), stride=1, padding=0,
                                groups=band_num, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.GConv2 = nn.Conv2d(outc, outc, (1, 1), stride=1,
                                padding=(0, 1), groups=band_num, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.ELU = nn.ELU(inplace=False)

    def forward(self, x, x_p, L):
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))
        y = torch.einsum('bijk,kp->bijp', (x, L))
        return self.ELU(torch.add(y, x_p))


class HGCN(nn.Module):
    def __init__(self, dim, chan_num, band_num):
        super().__init__()
        self.resGCN = resGCN(inc=dim * band_num, outc=dim * band_num,
                             band_num=band_num)
        self.ELU = nn.ELU(inplace=False)

    def forward(self, x, A_ds):
        L = torch.einsum('ik,kp->ip', (A_ds, torch.diag(
            torch.reciprocal(torch.sum(A_ds, dim=0)))))
        return self.resGCN(x, x, L).contiguous()


class MHGCN(nn.Module):
    def __init__(self, layers, dim, chan_num, band_num):
        super().__init__()
        self.chan_num = chan_num
        self.band_num = band_num
        self.A = torch.rand((1, chan_num * chan_num), dtype=torch.float32,
                            requires_grad=False)
        self.GATENet = GATENet(chan_num * chan_num, reduction_ratio=128)
        self.HGCN_layers = nn.ModuleList(
            [HGCN(dim=1, chan_num=chan_num, band_num=band_num)
             for _ in range(layers)])

    def forward(self, x):
        self.A = self.A.to(x.device)
        A_ds = self.GATENet(self.A).reshape(self.chan_num, self.chan_num)
        outputs = [x]
        for layer in self.HGCN_layers:
            x = layer(x, A_ds)
            outputs.append(x)
        return torch.cat(outputs, dim=1), A_ds


class Encoder(nn.Module):
    def __init__(self, chan_num, band_num, layers=2, hidden_2=64):
        super().__init__()
        self.chan_num = chan_num
        self.band_num = band_num
        cbam_ch = (layers + 1) * band_num
        ks = min(chan_num, 7)
        if ks % 2 == 0:
            ks -= 1  # ensure odd kernel size

        self.GGCN = MHGCN(layers=layers, dim=1, chan_num=chan_num,
                          band_num=band_num)
        self.CBAM = CBAMBlock(channel=cbam_ch, reduction=4, kernel_size=ks)
        self.fc1 = nn.Linear(chan_num * cbam_ch, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_2)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        # x: (B, C, T) → transpose → (B, T, C), unsqueeze → (B, T, 1, C)
        x = x.transpose(1, 2).unsqueeze(2)
        g_feat, _ = self.GGCN(x)
        g_feat = self.CBAM(g_feat)
        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        return out


# ==============================================================================
# Main NSAL-DGAT model — adapted for (B, C, T) input
# ==============================================================================

class NSAL_DGAT(nn.Module):
    """
    NSAL-DGAT emotion recognition model (Encoder + Classifier only).

    Args:
        n_channels:   number of EEG channels (32 for DEAP, 62 for SEED)
        n_timepoints: number of time points per sample (128 for DEAP)
        num_classes:  number of output classes
        layers:       number of HGCN layers (default 2)
        hidden_2:     feature dimension (default 64)
    """

    def __init__(self, n_channels, n_timepoints, num_classes,
                 layers=2, hidden_2=64):
        super().__init__()
        self.encoder = Encoder(chan_num=n_channels,
                               band_num=n_timepoints,
                               layers=layers,
                               hidden_2=hidden_2)
        self.classifier = nn.Linear(hidden_2, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        feat = self.encoder(x)          # (B, hidden_2)
        return self.classifier(feat)    # (B, num_classes)
