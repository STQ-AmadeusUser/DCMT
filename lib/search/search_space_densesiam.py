import torch.nn as nn

from .operations import OPS
from .search_space_base import Conv1_1_Block, Block
from .search_space_base import Network as BaseSearchSpace
from models.backbone.backbone import ResNet50Dilated
from models.neck.neck import ShrinkChannel
from models.match.match import ROIMatch
from models.head.head import Cls_Reg


class Network(BaseSearchSpace):
    def __init__(self, config):
        super(Network, self).__init__(config)

        self.backbone = ResNet50Dilated(affine=True)
        self.neck = ShrinkChannel(config.MODEL.NECK.IN_CHANNEL, config.MODEL.NECK.OUT_CHANNEL, affine=True)
        self.match = ROIMatch(config.MODEL.NECK.OUT_CHANNEL, 49, config.MODEL.NECK.OUT_CHANNEL, affine=True)

        self.fuse = nn.ModuleList()
        
        for i in range(self.num_blocks):
            input_config = self.input_configs[i]
            self.fuse.append(Block(
                input_config['in_chs'],
                input_config['ch'],
                input_config['strides'],
                input_config['num_stack_layers'],
                self.config
            ))

        self.collector = Conv1_1_Block(self.input_configs[-1]['in_chs'], config.SEARCH.LAST_DIM)
        self.prediction = Cls_Reg(config.SEARCH.LAST_DIM, affine=True)
        self.init_model(model_init=config.SEARCH.INIT.MODE)
        self.set_bn_param(config.SEARCH.INIT.BN_MOMENTUM, config.SEARCH.INIT.BN_EPS)
