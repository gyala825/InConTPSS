import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m, weight=None): # max_m 最大的margin值 default=0.5 weight 每个类别的权重
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)) # 计算每个类别的margin （m_list）
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.weight = weight #权重用于加权交叉熵损失

    def forward(self, x, target, mask):
        # [b*L, C] [b, L] [b, L]
        target = target.flatten()
        mask = mask.flatten()
        # index 将特定位置设为1
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        # 每个样本对应的margin
        batch_m = self.m_list[target].view(-1, 1)
        x_m = x - batch_m
        # 使用index选择是应用margin的样本x_m还是不应用margin的样本x
        # [b*L, C]
        output = torch.where(index, x_m, x)
        # 计算未减少的交叉熵损失
        unreduction_loss = F.cross_entropy(
            output, target, weight=self.weight, reduction="none"
        )
        if self.weight is not None:
            w = (self.weight[target] * mask).sum()
        else:
            w = float(mask.sum())
        # 计算损失 masked_select选择masked的损失值，然后求和除以权重
        loss = torch.sum(torch.masked_select(unreduction_loss, mask)) / w
        return loss
