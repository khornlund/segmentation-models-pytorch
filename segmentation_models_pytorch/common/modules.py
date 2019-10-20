import numpy as np

import torch
import torch.nn as nn


class TransposeUpsample(nn.Module):
    """Bilinear interpolation in space of scale.
    Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale
    Adapted from the CVPR'15 FCN code.
    See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        assert in_channels == out_channels
        if scale_factor != 1:
            assert scale_factor % 2 == 0, 'Scale should be even'
            self.in_channes = in_channels
            self.out_channels = out_channels
            self.scale_factor = int(scale_factor)
            self.padding = scale_factor // 2

            def upsample_filt(size):
                factor = (size + 1) // 2
                if size % 2 == 1:
                    center = factor - 1
                else:
                    center = factor - 0.5
                og = np.ogrid[:size, :size]
                return ((1 - abs(og[0] - center) / factor) *
                        (1 - abs(og[1] - center) / factor))

            kernel_size = scale_factor * 2
            bil_filt = upsample_filt(kernel_size)

            kernel = np.zeros(
                (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32
            )
            kernel[range(in_channels), range(out_channels), :, :] = bil_filt

            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                             stride=self.scale_factor, padding=self.padding)

            self.upconv.weight.data.copy_(torch.from_numpy(kernel))
            self.upconv.bias.data.fill_(0)
            self.upconv.weight.requires_grad = False
            self.upconv.bias.requires_grad = False
        else:
           self.scale_factor = 1

    def forward(self, x):
        if self.scale_factor == 1:
            return x
        else:
            return self.upconv(x)