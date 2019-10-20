import torch
import torch.nn as nn

from ..base.model import Model
from ..common.blocks import Conv2dReLU
from ..common.modules import TransposeUpsample


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(ConvBlock):

    def __init__(self, in_channels, skip_channels, out_channels,
                 use_batchnorm=True, scale_factor=2, scale_type="upsample"):
        super().__init__(in_channels + skip_channels, out_channels, use_batchnorm)

        middle_channels = skip_channels or out_channels
        self.layer1 = Conv2dReLU(in_channels, middle_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.layer2 = Conv2dReLU(middle_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

        if scale_type == "upsample":
            self.scale = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        elif scale_type == "transpose":
            self.scale = TransposeUpsample(in_channels, in_channels, scale_factor=scale_factor)
        else:
            raise ValueError

    def forward(self, x):
        x, skip = x
        x = self.scale(x)
        x = self.layer1(x)
        if skip is not None:
            x = x + skip
        x = self.layer2(x)
        return x


class AggregationBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2, scale_type="upsample"):
        super().__init__()
        if scale_type == "upsample":
            self.scale = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        elif scale_type == "transpose":
            self.scale = TransposeUpsample(in_channels, in_channels, scale_factor=scale_factor)
        else:
            raise ValueError

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.scale(x)
        x = self.conv(x)
        return x


class PANetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            block_type="upsample",
    ):
        super().__init__()

        if block_type not in ("upsample", "transpose"):
            raise ValueError("Supported block types: `upsample`, `transpose`")

        if center:
            channels = encoder_channels[0]
            self.center = ConvBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        # channels of input tensors for each decoder block
        in_channels = encoder_channels[:1] + decoder_channels[:-1]

        # channels of skip connection tensors for each decoder block
        # 0 channels for no skip connection
        skip_channels = encoder_channels[1:] + (0,)

        # output channels for each decoder block tensor
        out_channels = decoder_channels

        # number of channels for aggregation block
        agg_ch = out_channels[-2]

        # scale factors for aggregation pyramid
        scales = [2 ** i for i in range(len(out_channels[:-1]))][::-1]  # (8, 4, 2, 1)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(in_ch, sk_ch, out_ch, use_batchnorm, scale_factor=2, scale_type=block_type)
             for in_ch, sk_ch, out_ch in zip(in_channels[:-1], skip_channels[:-1], out_channels[:-1])]
        )

        self.aggregation_blocks = nn.ModuleList(
            [AggregationBlock(in_ch, agg_ch, scale_type=block_type, scale_factor=scale)
             for scale, in_ch in zip(scales, out_channels[:-1])]
        )

        self.final_upsample = DecoderBlock(in_channels[-1], 0, out_channels[-1],
                                           use_batchnorm, scale_type=block_type)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        # self.initialize()

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = encoder_head

        result = []
        for i, (decoder_block, aggregation_block) in enumerate(zip(self.decoder_blocks, self.aggregation_blocks)):
            x = decoder_block([x, skips[i]])
            y = aggregation_block(x)
            result.append(y)

        aggregation = sum(result)
        x = self.final_upsample([aggregation, None])  # no skip connection for last upsampling
        x = self.final_conv(x)

        return x
