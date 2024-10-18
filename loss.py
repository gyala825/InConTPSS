import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m, weight=None): 
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)) 
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.weight = weight 

    def forward(self, x, target, mask):
        target = target.flatten()
        mask = mask.flatten()
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        batch_m = self.m_list[target].view(-1, 1)
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        unreduction_loss = F.cross_entropy(
            output, target, weight=self.weight, reduction="none"
        )
        if self.weight is not None:
            w = (self.weight[target] * mask).sum()
        else:
            w = float(mask.sum())
        loss = torch.sum(torch.masked_select(unreduction_loss, mask)) / w
        return loss
