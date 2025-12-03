import torch
import torch.nn as nn
import torch.nn.functional as F

class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class WaveletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k_data = torch.tensor([[[[-1.0, 1.0], [-1.0, 1.0]]]])
        self.register_buffer('K', k_data)

    def forward(self, pred, target):
        p_f = F.conv2d(pred, self.K, stride=2)
        t_f = F.conv2d(target, self.K, stride=2)
        return F.l1_loss(p_f, t_f)