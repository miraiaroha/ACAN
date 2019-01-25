import functools
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules import *
from losses import *

def continuous2discrete(continuous_depth, Min, Max, num_classes):
    if continuous_depth.is_cuda:
        Min, Max, NC = Min.cuda(), Max.cuda(), num_classes.cuda()
    continuous_depth = torch.clamp(
        continuous_depth, Min.item(), Max.item())
    discrete_depth = torch.round(
        torch.log(continuous_depth / Min) / torch.log(Max / Min) * (NC - 1))
    return discrete_depth

def discrete2continuous(discrete_depth, Min, Max, num_classes):
    if discrete_depth.is_cuda:
        Min, Max, NC = Min.cuda(), Max.cuda(), num_classes.cuda()
    continuous_depth = torch.exp(
        discrete_depth / (NC - 1) * torch.log(Max / Min) + torch.log(Min))
    return continuous_depth

class SelfAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(SelfAttentionBlock_, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(True)
        )
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

    def forward(self, x):
        batch_size = x.size(0)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map


class PixelAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, scale=1):
        super(PixelAttentionBlock_, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_value = nn.Sequential(
            nn.Conv2d(in_channels, value_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(value_channels, value_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.pixel_att = SelfAttentionBlock_(in_channels, key_channels)
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
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        sim_map = self.pixel_att(x)
        value = self.f_value(x).view(
            batch_size, self.value_channels, -1).permute(0, 2, 1)
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        if self.scale > 1:
            context = F.upsample(input=context, size=(h, w),
                                 mode='bilinear', align_corners=True)
        return [context, sim_map]


class ContextAttentionModel(nn.Module):
    def __init__(self, block, layers, min_depth, max_depth, height, width, classifierType, scale=1, num_classes=32):
        self.inplanes = 64
        self.scale = scale
        self.min_depth = torch.tensor(min_depth, dtype=torch.float)
        self.max_depth = torch.tensor(max_depth, dtype=torch.float)
        self.num_classes = torch.tensor(num_classes, dtype=torch.float)
        # Note: classifierType: CE=Cross Entropy, OR=Ordinal Regression
        self.classifierType = classifierType
        super(ContextAttentionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # fixed the parameters before
        # for p in self.parameters():
        #     p.requires_grad = False
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4)
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False
        # extra added layers
        self.pixel_context = PixelAttentionBlock_(2048, 512, 2048, self.scale)
        self.image_context = nn.Sequential(
            nn.AvgPool2d((height // 8, width // 8), padding=0),
            nn.Dropout2d(0.5, inplace=True),
            Reshape(2048),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            Reshape(512, 1, 1),
            nn.Upsample(size=(height // 8, width // 8),
                        mode='bilinear', align_corners=True)
        )
        self.merge = nn.Sequential(
            nn.Dropout2d(0.5, inplace=True),
            nn.Conv2d(2048 + 512, 2048, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
            # nn.PixelShuffle(8),
        )
        if self.classifierType == 'CE':
            self.classifier = nn.Sequential(
                nn.Conv2d(2048, num_classes, kernel_size=1, stride=1),
                nn.Upsample(scale_factor=8, mode='bilinear',
                            align_corners=True)
            )
        else:  # 'OR'
            self.classifier = nn.Sequential(
                nn.Conv2d(2048, 2 * num_classes, kernel_size=1, stride=1),
                nn.Upsample(scale_factor=8, mode='bilinear',
                            align_corners=True)
            )
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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        context1, self.sim_map = self.pixel_context(x)
        context2 = self.image_context(x)
        self.context = context2
        x = torch.cat((context1, context2), dim=1)
        y = self.merge(x)
        y = self.classifier(y)
        if self.classifierType == 'OR':
            y = self.decode_ord(y)
        return y

    def get_sim_map(self):
        return self.sim_map

    def get_image_context(self):
        return self.context[:, :, 0, 0]

    def decode_ord(self, y):
        batch_size, prob, height, width = y.shape
        y = torch.reshape(y, (batch_size, self.num_classes, 2, height, width))
        denominator = torch.sum(torch.exp(y), 2)
        pred_score = torch.div(torch.exp(y[:, :, 1, :, :]), denominator)
        return pred_score

    def hard_cross_entropy(self, pred_score, Min, Max, num_classes):
        pred_label = torch.argmax(pred_score, 1, keepdim=True).float()
        pred_depth = discrete2continuous(pred_label, Min, Max, num_classes)
        return pred_depth

    def soft_cross_entropy(self, pred_score, Min, Max, num_classes):
        if pred_score.is_cuda:
            Min, Max, NC = Min.cuda(), Max.cuda(), num_classes.cuda()
        pred_prob = F.softmax(pred_score, dim=1)
        nc = torch.arange(NC.item())
        if pred_score.is_cuda:
            nc = nc.cuda()
        weight = nc * torch.log(Max / Min) / (NC - 1) + torch.log(Min)
        weight = weight.unsqueeze(-1)
        pred_prob = pred_prob.permute((0, 2, 3, 1))
        output = torch.exp(torch.matmul(pred_prob, weight))
        output = output.permute((0, 3, 1, 2))
        return output

    def hard_ordinal_regression(self, pred_prob, Min, Max, num_classes):
        mask = (pred_prob > 0.5).float()
        pred_label = torch.sum(mask, 1, keepdim=True)
        #pred_label = torch.round(torch.sum(pred_prob, 1, keepdim=True))
        pred_depth = (discrete2continuous(pred_label, Min, Max, num_classes) +
                      discrete2continuous(pred_label + 1, Min, Max, num_classes)) / 2
        return pred_depth

    def soft_ordinal_regression(self, pred_prob, Min, Max, num_classes):
        pred_prob_sum = torch.sum(pred_prob, 1, keepdim=True)
        Intergral = torch.floor(pred_prob_sum)
        Fraction = pred_prob_sum - Intergral
        depth_low = (discrete2continuous(Intergral, Min, Max, num_classes) +
                     discrete2continuous(Intergral + 1, Min, Max, num_classes)) / 2
        depth_high = (discrete2continuous(Intergral + 1, Min, Max, num_classes) +
                      discrete2continuous(Intergral + 2, Min, Max, num_classes)) / 2
        pred_depth = depth_low * (1 - Fraction) + depth_high * Fraction
        return pred_depth

    def inference(self, y):
        # mode
        # OR = Ordinal Regression
        # CE = Cross Entropy
        if self.classifierType == 'OR':
            if False:  # soft OR
                inferenceFunc = self.soft_ordinal_regression
            else:    # hard OR
                inferenceFunc = self.hard_ordinal_regression
        else:  # 'CE'
            if True:  # soft CE
                inferenceFunc = self.soft_cross_entropy
            else:     # hard CE
                inferenceFunc = self.hard_cross_entropy
        pred_depth = inferenceFunc(
            y, self.min_depth, self.max_depth, self.num_classes)
        return pred_depth

    class LossFunc(nn.Module):
        def __init__(self, min_depth, max_depth, num_classes, classifierType, scale=1, alpha=0, ignore_index=None):
            super(ContextAttentionModel.LossFunc, self).__init__()
            self.scale = scale
            self.min_depth = torch.tensor(min_depth, dtype=torch.float)
            self.max_depth = torch.tensor(max_depth, dtype=torch.float)
            self.num_classes = torch.tensor(num_classes, dtype=torch.float)
            self.ignore_index = ignore_index
            if classifierType == 'OR':
                self.AppearanceLoss = OrdinalRegression2d(
                    num_classes, ignore_index=ignore_index)
            else:  # 'CE'
                if True:
                    self.AppearanceLoss = CrossEntropy2d(
                        num_classes, ignore_index=ignore_index)
                else:  # 'Online Hard Example Mining'
                    if ignore_index == None:
                        self.AppearanceLoss = OhemCrossEntropy2d(
                            ignore_index=num_classes + 1)
                    else:
                        self.AppearanceLoss = OhemCrossEntropy2d(
                            ignore_index=ignore_index)
            self.KLDivLoss = KLDivergence()
            self.alpha = alpha

        def get_similarity(self, x, Max):
            if x.is_cuda:
                Max = Max.cuda()
            b, _, h, w = x.shape
            M = x.reshape((b, h * w, 1))
            N = x.reshape((b, 1, h * w))
            W = F.softmax(torch.log(Max) -
                          torch.abs(torch.log(M) - torch.log(N)), -1)
            W[torch.isnan(W)] = 0
            return W

        def get_attention_loss(self, sim_map, label):
            b, _, h, w = label.shape
            res_label = F.interpolate(
                label, size=(h // 8 // self.scale, w // 8 // self.scale), mode='nearest')
            gt_sim_map = self.get_similarity(res_label, self.max_depth)
            loss = self.KLDivLoss(sim_map, gt_sim_map)
            return loss

        def forward(self, pred_score, label, sim_map):
            """
                Args:
                    pred: [batch, num_classes, h, w]
                    label: [batch, 1, h, w]
            """
            label = continuous2discrete(
                label, self.min_depth, self.max_depth, self.num_classes)
            # image loss
            image_loss = self.AppearanceLoss(
                pred_score, label.squeeze(1).long())
            # attention loss
            att_loss = 0
            if self.alpha != 0:
                att_loss = self.get_attention_loss(sim_map, label)
                att_loss = self.alpha * att_loss
            # total loss
            total_loss = image_loss + att_loss
            return image_loss, att_loss, total_loss
