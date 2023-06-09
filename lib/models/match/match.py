import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from lib.models.activation import MishAuto as Mish


class ROIMatch(nn.Module):
    def __init__(self, in_dim=256, hid_dim=49, out_dim=256, affine=True, track_running_stats=True):
        super(ROIMatch, self).__init__()

        self.s_embed = nn.Conv2d(in_dim, in_dim, (1, 1))  # embedding for search feature
        self.t_embed = nn.Conv2d(in_dim, in_dim, (1, 1))  # embeeding for template feature

        self.roi_size, self.stride = 3, 8
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=(self.roi_size, self.roi_size), stride=(1, 1)),
            nn.BatchNorm2d(in_dim, affine=affine, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.1),
        )

        self.simple = nn.Sequential(
            nn.Conv2d(hid_dim, out_dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_dim, affine=affine, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.1),
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1)),
            # nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1))
        self.bias = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1)),
            # nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1))

    def forward(self, zf, xf, roi):

        xf = self.s_embed(xf)
        zf = self.t_embed(zf)

        B, C, Hx, Wx = xf.size()
        _, _, Hz, Wz = zf.size()

        # ROI Align Operation
        box_indices = torch.arange(B, dtype=torch.float32).reshape(-1, 1)  # box_indices(b,1)
        box_indices = torch.tensor(box_indices, dtype=torch.float32)
        batch_index = box_indices.to(roi.device)  # batch_index(b,1)
        batch_box = torch.cat((batch_index, roi), dim=1)  # batch_box(b,5)
        zf_3x3 = roi_align(zf, batch_box,
                           [self.roi_size, self.roi_size],
                           spatial_scale=1. / self.stride,
                           sampling_ratio=-1)  # zf_3x3(b,256,3,3)
        zf_1x1 = self.spatial_conv(zf_3x3)  # zf_1x1(b,256,1,1)

        # Pointwise-Add Module
        assert zf_1x1.shape[2] == 1
        zf_31x31 = zf_1x1.repeat(1, 1, Hx, Wx)  # zf_31x31(b,256,31,31)
        zf_add = zf_31x31.permute(0, 2, 3, 1).contiguous()  # zf_add(b,31,31,256)
        xf_add = xf.permute(0, 2, 3, 1).contiguous()  # xf_add(b,31,31,256)
        xf1 = xf_add + zf_add  # xf1(b,31,31,256)

        # Pairwise-Relation Module
        xf1 = xf1.view(B, -1, C)  # xf1(b,961,256)
        zf1 = zf.view(B, C, -1)  # zf1(b,256,49)
        xzf = torch.matmul(xf1, zf1)  # xzf(b,961,49)
        xzf = xzf.permute(0, 2, 1).contiguous().view(B, -1, Hx, Wx)  # xzf(b,49,31,31)

        # FiLM Module
        xzf1 = self.simple(xzf)  # xzf1(b,128,31,31)
        weight = self.weight(zf_1x1)  # weight(b,128,1,1)
        bias = self.bias(zf_1x1)  # bias(b,128,1,1)
        merge = (1 + weight) * xzf1 + bias  # merge(b,128,31,31)

        return merge
