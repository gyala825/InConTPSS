import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class stem(nn.Module):
    def __init__(self, channels, dropout):
        super(stem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Conv1d(channels, channels, 3, padding=1)
        )
        self.bn = nn.BatchNorm1d(channels)
        self.branch_1 = nn.Sequential(
            BasicConv1d(channels, channels // 2, 3),
            BasicConv1d(channels // 2, channels // 2, 3),
        )
        self.branch_2 = nn.Sequential(
            BasicConv1d(channels, channels // 2, 3),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.hardswish(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        out = torch.cat([x1, x2], dim=1)
        out += x
        out = self.dropout(out)
        return out




