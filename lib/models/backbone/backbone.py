import math
import torch
import torch.nn as nn
import numpy as np

from lib.models.backbone.modules import Bottleneck
from lib.models.activation import MishAuto as Mish


class ResNet50Dilated(nn.Module):
    '''
    modified ResNet50 with dialation conv in stage3 and stage4
    used in SiamRPN++/Ocean/OceanPLus/AutoMatch
    '''
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], affine=True, track_running_stats=True):
        super(ResNet50Dilated, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine, track_running_stats=track_running_stats)
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], affine=affine, track_running_stats=track_running_stats)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, affine=affine, track_running_stats=track_running_stats)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, affine=affine, track_running_stats=track_running_stats)
        self.feature_size = (256 + 128) * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, update=False, affine=True, track_running_stats=True):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, affine=affine, track_running_stats=track_running_stats),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion, affine=affine, track_running_stats=track_running_stats),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, dilation=dilation,
                            affine=affine, track_running_stats=track_running_stats))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                affine=affine, track_running_stats=track_running_stats))

        if update:
            self.inplanes = int(self.inplanes / 2)  # for online
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.act(x)
        x = self.maxpool(x_)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)

        return p3
