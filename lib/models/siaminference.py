''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: general SOT pipeline, support SiamFC, SiamDW, Ocean, AutoMatch
Data: 2021.6.23
'''

import torch
import importlib
import torch.nn as nn


class SiamInference(nn.Module):
    def __init__(self, archs=None):
        super(SiamInference, self).__init__()
        self.cfg = archs['cfg']
        self.init_arch(archs)
        self.init_hyper()
        self.init_loss()

    def init_arch(self, inputs):
        self.backbone = inputs['backbone']
        self.neck = inputs['neck']
        self.head = inputs['head']

    def init_hyper(self):
        self.lambda_u = 0.1
        self.lambda_s = 0.2
        # self.grids()

    def init_loss(self):
        if self.cfg is None:
            raise Exception('Not set config!')

        loss_module = importlib.import_module('models.loss')
        cls_loss_type = self.cfg.MODEL.LOSS.CLS_LOSS
        reg_loss_type = self.cfg.MODEL.LOSS.REG_LOSS

        self.cls_loss = getattr(loss_module, cls_loss_type)
        self.reg_loss = getattr(loss_module, reg_loss_type)

    def forward(self, inputs):
        """
        inputs:
         - template: BCHW, H*W:127*127
         - search: BCHW, H*W:255*255
         - cls_label: BH'W' or B2H'W'
         - reg_label: B4H'W (optional)
         - reg_weight: BH'W' (optional)
        """

        template, search = inputs['template'], inputs['search']

        # backbone
        zfs = self.backbone(template)  # zfs.keys(): dict_keys(['l1', 'p1', 'p2', 'p3'])
        # zfs['l1']: bx64x61x61    zfs['p1']: bx256x31x31    zfs['p2']: bx512x15x15    zfs['p3']: bx1024x15x15
        xfs = self.backbone(search)  # xfs.keys(): dict_keys(['l1', 'p1', 'p2', 'p3'])
        # xfs['l1']: bx64x125x125    xfs['p1']: bx256x63x63    xfs['p2']: bx512x31x31    xfs['p3']: bx1024x31x31
        zf, xf = zfs['p3'], xfs['p3']  # zf(b,1024,15,15), xf(b,1024,31,31)

        # neck
        xf_neck = self.neck(xf, crop=False)
        zf_neck = self.neck(zf, crop=True)
        zf, xf = zf_neck['crop'], xf_neck['ori']  # zf(b,1024,7,7), xf(b,1024,31,31)

        # head
        head_inputs = {'xf': xf, 'zf': zf, 'target_box': inputs['template_bbox'],}
        preds = self.head(head_inputs)

        # loss
        cls_label, reg_label, reg_weight = inputs['cls_label'], inputs['reg_label'], inputs['reg_weight']
        cls_pred, reg_pred = preds['cls'], preds['reg']
        reg_loss = self.reg_loss(reg_pred, reg_label, reg_weight)
        cls_loss = self.cls_loss(cls_pred, cls_label)
        loss = {'cls_loss': cls_loss, 'reg_loss': reg_loss}

        return loss

    # only for testing
    def template(self, inputs):
        """
        inputs:
         - template: BCHW, H*W:127*127
         - template_mask: BHW (optional)
        """

        template = inputs['template']
        zfs = self.backbone(template)
        zf = zfs['p3']

        zf_neck = self.neck(zf, crop=True)
        self.zf = zf_neck['crop']

        if 'template_mask' in inputs.keys():
            self.template_mask = inputs['template_mask'].float()

        if 'target_box' in inputs.keys():
            self.target_box = torch.tensor(inputs['target_box'], dtype=torch.float32).to(self.zf.device)
            self.target_box = self.target_box.view(1, 4)

    def track(self, inputs):
        """
        inputs:
         - search: BCHW, H*W:255*255
        """

        search = inputs['search']
        xfs = self.backbone(search)
        xf = xfs['p3']

        xf_neck = self.neck(xf, crop=False)
        xf = xf_neck['ori']
        head_inputs = {'xf': xf, 'zf': self.zf, 'target_box': self.target_box, }
        preds = self.head(head_inputs)

        return preds
