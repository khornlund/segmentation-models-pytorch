import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.model import Model
from ..common.activations import Swish
from ..common.blocks import Conv2dWS


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, weight_std=False, upsample=False, activation='relu'):

        if weight_std:
            Conv2d = Conv2dWS
        else:
            Conv2d = nn.Conv2d

        if activation == 'relu':
            relu_fn = nn.ReLU(inplace=True)
        elif activation == 'swish':
            relu_fn = Swish()
        else:
            raise ValueError(f'`activation` must be "relu" or "swish"')

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            Conv2d(in_channels, out_channels, (3, 3),
                              stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            relu_fn,
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x



class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, weight_std, n_upsamples=0, activation='relu'):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples), activation=activation)
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(
                    out_channels, out_channels, weight_std, upsample=True, activation=activation))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class FPNDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_upsampling=4,
            final_channels=1,
            dropout=0.2,
            weight_std=False,
            merge_policy='add',
            activation='relu'
    ):
        super().__init__()

        if merge_policy not in ['add', 'cat']:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(merge_policy))
        self.merge_policy = merge_policy

        self.final_upsampling = final_upsampling
        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, weight_std, 3, activation)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, weight_std, 2, activation)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, weight_std, 1, activation)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, weight_std, 0, activation)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        if self.merge_policy == 'cat':
            segmentation_channels *= 4

        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def forward(self, x):
        c5, c4, c3, c2, _ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        if self.merge_policy == 'add':
            x = s5 + s4 + s3 + s2
        elif self.merge_policy == 'cat':
            x = torch.cat([s5, s4, s3, s2], dim=1)

        x = self.dropout(x)
        x = self.final_conv(x)

        if self.final_upsampling is not None and self.final_upsampling > 1:
            x = F.interpolate(x, scale_factor=self.final_upsampling, mode='bilinear', align_corners=True)
        return x
