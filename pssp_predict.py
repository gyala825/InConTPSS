import torch
from torch import nn
from net.hybrid_net import pssp_inception
from net.repr_net import pssp_repr
import torch.nn.functional as F
   
class Normalized_FC(nn.Module):
    def __init__(self, num_features, num_classes, tau=16):
        super(Normalized_FC, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.tau = tau

    def forward(self, x):
        out = F.linear(F.normalize(x), self.tau*F.normalize(self.weight))
        return out 

class SS_predict(nn.Module):
    def __init__(self, hybrid_features, embed_features, num_class, depth):
        super(SS_predict, self).__init__()
        hybrid_channels = 4 * (hybrid_features // 4)
        self.hybrid_net = pssp_inception(hybrid_features, hybrid_channels, depth)
        self.embed_net = pssp_repr(embed_features + hybrid_channels)
        self.fc = nn.Linear(32, num_class, bias=False)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, x1, x2):
        
        hybrid_out = self.hybrid_net(x1)
        output = self.embed_net(torch.cat([hybrid_out, x2], dim=1))
        output = self.bn(output)
        emb_out = output
        out = self.fc((output.transpose(1,2)))
        return out
    
    
class non_SS_predict(nn.Module):
    def __init__(self, hybrid_features, embed_features, num_class, depth):
        super(non_SS_predict, self).__init__()
        hybrid_channels = 4 * (hybrid_features // 4)
        self.conv1= nn.Conv1d(embed_features+hybrid_features, 96, 1)
        self.hy_net = pssp_inception(hybrid_features, hybrid_channels ,depth)
        self.hy_net1 = pssp_repr(hybrid_channels)
        self.fc = nn.Linear(96, 32, bias=False)
        self.fc1 = nn.Linear(32, num_class, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, x1, x2): 
        # x = torch.cat([x1, x2], dim=1)
        x = self.hy_net1(self.hy_net(x1))
        # x = self.conv1(x)
        x = self.bn(x)
        x = x.transpose(1,2)
        # x = self.fc(x)
        output = self.fc1(x)
        return output
        
  
  

if __name__ == '__main__':
    model = non_SS_predict(50, 1280, 8, 2)
    input_data2 = torch.randn(32, 1280, 20)
    input_data1 = torch.randn(32, 50, 20)
    out = model(input_data1, input_data2)
    print(out.shape) 

