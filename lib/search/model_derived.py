import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations import OPS
import utils.search_helper as searcher
from lib.models.backbone.backbone import ResNet50Dilated
from lib.models.neck.neck import ShrinkChannel
from lib.models.match.match import ROIMatch
from lib.models.head.head import Cls_Reg
from lib.models.activation import MishAuto as Mish


class Block(nn.Module):

    def __init__(self, in_ch, block_ch, head_op, stack_ops, stride):
        super(Block, self).__init__()
        self.head_layer = OPS[head_op](in_ch, block_ch, stride, affine=True, track_running_stats=True)

        modules = []
        for stack_op in stack_ops:
            modules.append(OPS[stack_op](block_ch, block_ch, 1, affine=True, track_running_stats=True))
        self.stack_layers = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.head_layer(x)
        x = self.stack_layers(x)
        return x


class Conv1_1_Block(nn.Module):

    def __init__(self, in_ch, block_ch):
        super(Conv1_1_Block, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=block_ch, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(block_ch),
            nn.LeakyReLU(0.0)
        )

    def forward(self, x):
        return self.conv1_1(x)


class DENSESIAM_Net(nn.Module):
    def __init__(self, net_config, config=None):
        """
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        """
        super(DENSESIAM_Net, self).__init__()
        self.config = config
        self.net_config = searcher.parse_net_config(net_config)

        backbone_cfg = self.net_config.pop(0)
        self.backbone = ResNet50Dilated(affine=True, track_running_stats=True)
        neck_cfg = self.net_config.pop(0)
        self.neck = ShrinkChannel(neck_cfg[0][0], neck_cfg[0][1], affine=True, track_running_stats=True)
        match_cfg = self.net_config.pop(0)
        self.match = ROIMatch(match_cfg[0][0], 49, match_cfg[0][1], affine=True, track_running_stats=True)

        self.fuse = nn.ModuleList()
        for config in self.net_config:
            if config[1] == 'collector':
                continue
            if config[1] == 'prediction':
                continue
            self.fuse.append(Block(config[0][0], config[0][1], config[1], config[2], config[-1]))
            # self.fuse.append(Block(256, 256, 'wavemlp_k7_t4', ['wavemlp_k7_t4', 'wavemlp_k7_t4'], 1))

        block_last_dim = self.net_config[-2][0][0]
        last_dim = self.net_config[-2][0][1]
        self.collector = Conv1_1_Block(block_last_dim, last_dim)
        self.prediction = Cls_Reg(last_dim, affine=True)

        # self.init_model()
        self.set_bn_param(0.1, 0.00001)

    def forward(self, z, x, bbox):
        zf = self.backbone(z)
        xf = self.backbone(x)
        zf = self.neck(zf)
        xf = self.neck(xf)
        block_data = self.match(zf, xf, bbox)

        if len(self.fuse) > 0:
            for i, block in enumerate(self.fuse):
                block_data = block(block_data)

        block_data = self.collector(block_data)
        cls, reg = self.prediction(block_data)

        return cls, reg

    def init_model(self, model_init='he_fout', init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return
