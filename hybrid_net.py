import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import BasicConv1d
from tcn import TemporalConvNet

def norm(x):
    out = F.normalize(x - x.mean(dim=-1, keepdim=True), 2, dim=-1)
    return out

class stem(nn.Module):
    def __init__(self, channels, dropout):
        super(stem, self).__init__()
        self.conv = nn.Sequential(
            BasicConv1d(channels, channels, 1),
            BasicConv1d(channels, channels, 3)
        )
        self.bn = nn.BatchNorm1d(channels)
        self.branch_1 = nn.Sequential(
            BasicConv1d(channels, channels // 2, 3),
            BasicConv1d(channels // 2, channels // 2, 3),
        )
        self.branch_2 = nn.Sequential(
            BasicConv1d(channels, channels // 2, 1),
            TemporalConvNet(channels // 2, [channels // 4, channels // 8]),
            BasicConv1d(channels // 8, channels // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = F.hardswish(out)
        x1 = self.branch_1(out)
        x2 = self.branch_2(out)
        outputs = F.hardswish(self.bn(torch.cat([x1, x2], dim=1)))
        outputs += identity
        outputs = self.dropout(outputs)
        return outputs


class pssp_inception(nn.Module):
    def __init__(self, num_features, num_channels, depth, dropout=0.2):
        super(pssp_inception, self).__init__()   
        self.conv = nn.Conv1d(num_features, num_channels, 1)
        self.stem = stem(num_channels, dropout)
        self.inception = nn.ModuleList(
            [inception_block(num_channels, dropout) for _ in range(depth)]
        )  
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.ca = CoordAtt(num_channels)
    
    def forward(self, x):
        x = self.conv(self.dropout(x))
        x = self.stem(x)
        for i in range(self.depth):
            x = self.inception[i](x)
        return x

    
class inception_block(nn.Module):
    def __init__(self, channels, dropout):
        super(inception_block, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Conv1d(channels, channels, 1),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        self.branch_3 = nn.Sequential(              
            TemporalConvNet(channels, [channels // 2, channels // 4]),
            nn.Conv1d(channels // 4, channels, 1),
        )   
        self.branch_4 = nn.Sequential(              
            TemporalConvNet(channels, [channels // 2, channels // 4, channels // 8]),
            nn.Conv1d(channels // 8, channels, 1),
        )    
        self.conv1 = nn.Conv1d(channels * 4, 128, 1)
        self.conv2 = nn.Conv1d(128, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.ca = CoordAtt(channels)
   
    def forward(self, x):
        identity = x
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        out = self.conv2(self.conv1(torch.cat([x1, x2, x3, x4], 1)))
        out = F.hardswish(self.bn(out))
        out += identity
        out = self.dropout(out)
        return out

