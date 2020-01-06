##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import OrderedDict

class Reshape(nn.Module):
    # input the dimensions except the batch_size dimension
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)

class PixelAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(PixelAttentionBlock_, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_key = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0)),
            ('bn',   nn.BatchNorm2d(key_channels)),
            ('relu', nn.ReLU(True))]))
        self.parameter_initialization()
        self.f_query = self.f_key

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_att(self, x):
        batch_size = x.size(0)
        query = self.f_query(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map

    def forward(self, x):
        raise NotImplementedError


class SelfAttentionBlock_(PixelAttentionBlock_):
    def __init__(self, in_channels, key_channels, value_channels, scale=1):
        super(SelfAttentionBlock_, self).__init__(in_channels, key_channels)
        self.scale = scale
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        kernel_size = 3
        self.f_value = nn.Sequential(OrderedDict([
            ('conv1',  nn.Conv2d(in_channels, value_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)),
            ('relu1',  nn.ReLU(inplace=True)),
            ('conv2',  nn.Conv2d(value_channels, value_channels, kernel_size=1, stride=1)),
            ('relu2',  nn.ReLU(inplace=True))]))
        self.parameter_initialization()

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size, c, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)
        sim_map = self.forward_att(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1).permute(0, 2, 1)
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        return [context, sim_map]

class SADecoder(nn.Module):
    def __init__(self, in_channels=2048, key_channels=512, value_channels=2048, height=224, width=304):
        super(SADecoder, self).__init__()
        out_channels = 512
        self.saconv = SelfAttentionBlock_(in_channels, key_channels, value_channels)
        self.image_context = nn.Sequential(OrderedDict([
            ('avgpool', nn.AvgPool2d((height // 8, width // 8), padding=0)),
            ('dropout', nn.Dropout2d(0.5, inplace=True)),
            ('reshape1', Reshape(2048)),
            ('linear1', nn.Linear(2048, 512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(512, 512)),
            ('relu2', nn.ReLU(inplace=True)),
            ('reshape2', Reshape(512, 1, 1)),
            ('upsample', nn.Upsample(size=(height // 8, width // 8), mode='bilinear', align_corners=True))]))
        self.merge = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout2d(0.5, inplace=True)),
            ('conv1',    nn.Conv2d(value_channels+out_channels, value_channels, kernel_size=1, stride=1)),
            ('relu',     nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout2d(0.5, inplace=False))]))

    def get_sim_map(self):
        return self.sim_map

    def forward(self, x):
        context1, sim_map = self.saconv(x)
        self.sim_map = sim_map
        context2 = self.image_context(x)
        x = torch.cat((context1, context2), dim=1)
        x = self.merge(x)
        return x, self.sim_map
