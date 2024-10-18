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

    
class ECANet1d(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECANet1d, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        # AdaptiveAvgPool(out_size)
        # AdaptiveAvgPool(out_size)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.b = b

    def forward(self, x):
        avg_y = self.avg_pool(x)
        # max_y = self.max_pool(x)
        avg_out = self.conv(avg_y.transpose(-1, -2)).transpose(-1, -2).sigmoid()
        # max_out = self.conv(max_y.transpose(-1, -2)).transpose(-1, -2)
        out = self.gamma * avg_out + self.b
        return x * out

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
        # self.branch_eca = ECANet1d(channels // 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.hardswish(x)
        x1 = self.branch_1(x)
        # x1 = self.branch_eca(x1)
        x2 = self.branch_2(x)
        # x2 = self.branch_eca(x2)
        out = torch.cat([x1, x2], dim=1)
        out += x
        # out = F.hardswish(out) 
        out = self.dropout(out)
        return out


'''class stem(nn.Module):
    def __init__(self, channels, dropout):
        super(stem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Conv1d(channels, channels, 3, padding=1)
        )
        self.bn = nn.BatchNorm1d(channels)
        self.branch_1 = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 3, padding=1),
            nn.Conv1d(channels // 2, channels // 2, 3, padding=1)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 3, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.bn(self.conv(x))
        y = F.hardswish(x)
        # x = x * y
        x1 = self.branch_1(x)
        
        x2 = self.branch_2(x)
        out = torch.cat([x1, x2], dim=1)
        out += x
        out = self.dropout(out)
        return out  ''' 
    
class eca_layer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(eca_layer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma)) 
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
 
   
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, L = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)