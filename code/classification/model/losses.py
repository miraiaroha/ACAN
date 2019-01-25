import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def safe_log(self, x):
        return torch.log(torch.clamp(x, 1e-8, 1e8))

    def forward(self, Q, P):
        """
            Args:
                P: ground truth probability distribution [batch, n, n]
                Q: predicted probability distribution [batch, n, n]
        """
        kl_loss = P * self.safe_log(P / Q)
        pixel_loss = torch.sum(kl_loss, dim=-1)
        total_loss = torch.mean(pixel_loss)
        return total_loss

class OrdinalRegression2d(nn.Module):
    def __init__(self, num_classes, ignore_index=0):
        super(OrdinalRegression2d, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def safe_log(self, x):
        return torch.log(torch.clamp(x, 1e-8, 1e8))

    def forward(self, pred, label):
        """
            Args:
                pred: [batch, num_classes, h, w]
                label: [batch, h, w]
        """
        if self.ignore_index != None:
            mask = (label != self.ignore_index).float()
        else:
            mask = torch.ones_like(label, dtype=torch.float)
        label = label.unsqueeze(3).long()
        pred = pred.permute(0, 2, 3, 1)
        mask10 = (torch.arange(self.num_classes)).cuda() < label
        mask01 = (torch.arange(self.num_classes)).cuda() >= label
        mask10 = mask10.float()
        mask01 = mask01.float()
        entropy = self.safe_log(pred) * mask10 + \
            self.safe_log(1 - pred) * mask01
        pixel_loss = -torch.sum(entropy, -1)
        masked_pixel_loss = pixel_loss * mask
        total_loss = torch.sum(masked_pixel_loss) / mask.sum()
        return total_loss

class CrossEntropy2d(nn.Module):
    def __init__(self, num_classes, ignore_index=None):
        super(CrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def safe_log(self, x):
        return torch.log(torch.clamp(x, 1e-8, 1e8))

    def forward(self, pred, label):
        """
            Args:
                pred: [batch, num_classes, h, w]
                label: [batch, h, w]
        """
        if self.ignore_index != None:
            mask = (label != self.ignore_index).float()
        else:
            mask = torch.ones_like(label, dtype=torch.float)
        label = label.unsqueeze(3).long()
        pred = pred.permute(0, 2, 3, 1)
        one_hot_label = (torch.arange(self.num_classes)).cuda() == label
        one_hot_label = one_hot_label.float()
        entropy = one_hot_label * self.safe_log(pred) + \
            (1 - one_hot_label) * self.safe_log(1 - pred)
        pixel_loss = - torch.sum(entropy, -1)
        masked_pixel_loss = pixel_loss * mask
        total_loss = torch.sum(masked_pixel_loss) / mask.sum()
        return total_loss

