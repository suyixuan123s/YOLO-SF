import numpy as np
import torch
import torch.nn as nn
import math

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class ELANB(nn.Module):
    # ELAN__ Blocks_
    def __init__(self, in_dims, out_dims, expand_ratio=0.5):
        super(ELANB, self).__init__()
        ind_dims = int(in_dims * expand_ratio)

        self.cv1 = Conv(in_dims, ind_dims, k=1)
        # self.cv1 = DWConv(in_dims, ind_dims, k=1)
        self.cv2 = Conv(in_dims, ind_dims, k=1)
        self.cv3 = nn.Sequential(*[
            Conv(ind_dims, ind_dims, k=3, p=1)
            for _ in range(2)
        ])
        self.cv4 = nn.Sequential(*[
            Conv(ind_dims, ind_dims, k=3, p=1)
            for _ in range(2)
        ])

        self.out = Conv(ind_dims*4, out_dims, k=1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        y1 = self.cv3(x2)
        y2 = self.cv4(y1)

        out = self.out(torch.cat([x1, x2, y1, y2], dim=1))

        return out
