import torch
from torch import nn
from hybrid_net import pssp_inception
from repr_net import pssp_repr
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
   
    


    print(out.shape) 

