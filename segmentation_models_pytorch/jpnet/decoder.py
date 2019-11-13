import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU, SCSEModule, SeparableConv2d
from ..base.model import Model


class JPU(nn.Module):
    def __init__(self, encoder_channels, width=512, up_kwargs=None):
        super().__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(encoder_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(encoder_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        self.dilation1 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.dilation4 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3, padding=8, dilation=8, bias=False
            ),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [self.conv5(x[-1]), self.conv4(x[-2]), self.conv3(x[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat(
            [
                self.dilation1(feat),
                self.dilation2(feat),
                self.dilation3(feat),
                self.dilation4(feat),
            ],
            dim=1,
        )

        return x[0], x[1], x[2], feat


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=atrous_rate,
            dilation=atrous_rate,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )
    return block


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), **self._up_kwargs)


class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = AsppPooling(in_channels, out_channels, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False),
        )

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)

        return self.project(y)


class ASPPHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up_kwargs,
        atrous_rates=(12, 24, 36),
        dropout=0.1,
    ):
        super().__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPP_Module(in_channels, atrous_rates, up_kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout2d(dropout, False),
            nn.Conv2d(inter_channels, out_channels, 1),
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x


class JPNDecoder(Model):
    def __init__(
        self,
        encoder_channels,
        width=512,
        final_channels=1,
        up_kwargs={"mode": "bilinear", "align_corners": True},
        dropout=0.1,
    ):
        super().__init__()
        self._up_kwargs = up_kwargs
        self.jpu = JPU(encoder_channels, width, self._up_kwargs)
        self.head = ASPPHead(
            width * 4, final_channels, self._up_kwargs, dropout=dropout
        )
        self.initialize()

    def forward(self, x):
        _, _, h, w = x[-1].size()
        _, _, _, c4 = self.jpu(x)
        x = self.head(c4)
        x = F.interpolate(x, (h * 2, w * 2), **self._up_kwargs)
        return x
