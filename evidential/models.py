import math

import torch
import torch.nn as nn


class IntraViewAAModule(nn.Module):
    def __init__(self):
        super(IntraViewAAModule, self).__init__()
        base_filter = 8
        self.deformconv0 = deformconvgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)
        self.conv0 = convgnrelu(base_filter * 4, base_filter * 2, kernel_size=1, stride=1, dilation=1)
        self.deformconv1 = deformconvgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)
        self.conv1 = convgnrelu(base_filter * 4, base_filter * 1, kernel_size=1, stride=1, dilation=1)
        self.deformconv2 = deformconvgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)
        self.conv2 = convgnrelu(base_filter * 4, base_filter * 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x0, x1, x2):
        m0 = self.conv0(self.deformconv0(x0))
        x1_ = self.conv1(self.deformconv1(x1))
        x2_ = self.conv2(self.deformconv2(x2))
        m1 = nn.functional.interpolate(x1_, scale_factor=2, mode='bilinear', align_corners=True)
        m2 = nn.functional.interpolate(x2_, scale_factor=4, mode='bilinear', align_corners=True)
        return torch.cat([m0, m1, m2], 1)