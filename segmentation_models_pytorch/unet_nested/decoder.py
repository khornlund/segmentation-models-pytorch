import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU, SCSEModule
from ..base.model import Model


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class NestedUNetDecoder(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py
    """
    def __init__(
        self,
        encoder_channels,
        decoder_channels=(64, 128, 256, 512),
        out_ch=1,
        deep_supervision=False
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # parameterise decoder channels:
        # if the same encoder is used as original code, channels will be the same
        # however many encoders like efficientnet have smaller output shapes, so we want to
        # be able to increase the decoder channels rather than use the encoder out shapes
        e0, e1, e2, e3, e4 = list(reversed(encoder_channels))
        d0, d1, d2, d3 = decoder_channels

        self.conv0_1 = conv_block_nested(e0 + e1, d0)
        self.conv1_1 = conv_block_nested(e1 + e2, d1)
        self.conv2_1 = conv_block_nested(e2 + e3, d2)
        self.conv3_1 = conv_block_nested(e3 + e4, d3)

        self.conv0_2 = conv_block_nested(e0 + d0 + d1, d0)
        self.conv1_2 = conv_block_nested(e1 + d1 + d2, d1)
        self.conv2_2 = conv_block_nested(e2 + d2 + d3, d2)

        self.conv0_3 = conv_block_nested(e0 + d0 + d0 + d1, d0)
        self.conv1_3 = conv_block_nested(e1 + d1 + d1 + d2, d1)

        self.conv0_4 = conv_block_nested(e0 + d0 + d0 + d0 + d1, d0)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(d0, out_ch, kernel_size=1)
            self.final2 = nn.Conv2d(d0, out_ch, kernel_size=1)
            self.final3 = nn.Conv2d(d0, out_ch, kernel_size=1)
            self.final4 = nn.Conv2d(d0, out_ch, kernel_size=1)
        else:
            self.final = nn.Conv2d(d0, out_ch, kernel_size=1)

    def features(self, x):
        x4_0, x3_0, x2_0, x1_0, x0_0 = x

        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        return [x0_4, x0_3, x0_2, x0_1]

    def forward(self, x):
        x0_4, x0_3, x0_2, x0_1 = self.features(x)

        if self.deep_supervision:
            o1 = self.Up(self.final1(x0_1))
            o2 = self.Up(self.final1(x0_2))
            o3 = self.Up(self.final1(x0_3))
            o4 = self.Up(self.final1(x0_4))
            return [o4, o3, o2, o1]
        else:
            return self.Up(self.final(x0_4))
