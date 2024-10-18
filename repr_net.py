import torch
import torch.nn as nn
import torch.nn.functional as F
from net.net_utils import BasicConv1d
from net.CA import CoordAtt
from net.tcn import TemporalConvNet
    
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
        # self.bn = nn.BatchNorm1d(channels)   #多余 pth删参数
        self.bn1 = nn.BatchNorm1d(channels)
        self.branch_1 = nn.Sequential(
            BasicConv1d(channels, channels // 2, 3),
            BasicConv1d(channels // 2, channels // 2, 3),
        )
        self.branch_2 = nn.Sequential(
            BasicConv1d(channels, channels // 2, 1),
            TemporalConvNet(channels // 2, [channels // 4, channels // 8]),
            BasicConv1d(channels // 8, channels // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        identity = x 
        out = self.conv(x)
        x1 = self.branch_1(out)
        x2 = self.branch_2(out)
        outputs = F.hardswish(self.bn1(torch.cat([x1, x2], dim=1)))
        outputs += identity
        outputs = self.dropout(outputs)
        return outputs


class pssp_repr(nn.Module):
    def __init__(self, num_channels, dropout=0.2):
        super(pssp_repr, self).__init__()
        self.stem = stem(num_channels, dropout)
        self.conv = BasicConv1d(num_channels, 96, kernel_size=9)
        self.fc = nn.Linear(96, 32, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.ca = CoordAtt(num_channels) 
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.stem(x)
        x = self.conv(x)    
        x = x.transpose(1,2)
        x = self.fc(x)
        return x.transpose(1,2)
#     x [b, 32, L]
    

if __name__ == '__main__':
    model = pssp_repr(1280)
    input_data = torch.randn(32, 1280, 20)
    out = model(input_data)
    print(out.shape)