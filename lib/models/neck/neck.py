import torch.nn as nn


class ShrinkChannel(nn.Module):
    '''
    shrink feature channel
    '''
    def __init__(self, in_channels, out_channels, affine=True, track_running_stats=True):
        super(ShrinkChannel, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=track_running_stats),
            )

    def forward(self, x):
        x_ori = self.downsample(x)
        if x_ori.size(3) < 20:
            l = 4
            r = -4
            xf = x_ori[:, :, l:r, l:r]
            return xf
        else:
            return x_ori
