import re
import math
import collections
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo


class EfficientNetEncoder(nn.Module):
    """
    Implementation taken from: https://github.com/lukemelas/EfficientNet-PyTorch
    """

    def __init__(self,
        width_coeff,
        depth_coeff,
        image_size,
        dropout_rate,
        drop_connect_rate,
        block_chunks
    ):
        super().__init__()
        self._blocks_args = self.get_block_args()
        self._global_params = get_global_params(width_coeff, depth_coeff, image_size,
                                                dropout_rate, drop_connect_rate)
        self.block_chunks = block_chunks

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        n_blocks = len(self._blocks)
        assert self.block_chunks[-1] == n_blocks, f'{self.block_chunks[-1]}, {n_blocks}'
        self._out_shapes = [self._bn0.num_features]
        self._out_shapes += [self._blocks[i - 1]._bn2.num_features for i in self.block_chunks[1:]]
        self._out_shapes = list(reversed(self._out_shapes))
        # precalc drop connect rates
        self.drop_connect_rates = np.arange(0, n_blocks, dtype=np.float) / n_blocks
        self.drop_connect_rates *= self._global_params.drop_connect_rate

    def forward_blocks(self, x, start_idx, end_idx):
        for idx in range(start_idx, end_idx):
            x = self._blocks[idx](x, self.drop_connect_rates[idx])
        return x

    def forward(self, x):
        x0 = relu_fn(self._bn0(self._conv_stem(x)))
        x1 = self.forward_blocks(x0, self.block_chunks[0], self.block_chunks[1])
        x2 = self.forward_blocks(x1, self.block_chunks[1], self.block_chunks[2])
        x3 = self.forward_blocks(x2, self.block_chunks[2], self.block_chunks[3])
        x4 = self.forward_blocks(x3, self.block_chunks[3], self.block_chunks[4])
        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('_conv_head.weight')
        state_dict.pop('_bn1.weight')
        state_dict.pop('_bn1.bias')
        state_dict.pop('_bn1.running_mean')
        state_dict.pop('_bn1.running_var')
        state_dict.pop('_bn1.num_batches_tracked')
        state_dict.pop('_fc.bias')
        state_dict.pop('_fc.weight')
        super().load_state_dict(state_dict, **kwargs)

    def get_block_args(self):
        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25',
            'r2_k3_s22_e6_i16_o24_se0.25',
            'r2_k5_s22_e6_i24_o40_se0.25',
            'r3_k3_s22_e6_i40_o80_se0.25',
            'r3_k5_s11_e6_i80_o112_se0.25',
            'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25',
        ]
        blocks_args = BlockDecoder.decode(blocks_args)
        return blocks_args

    def info(self):
        msg = '== EfficientNetEncoder ==\n\n'
        msg += 'x0: conv_stem\n'
        chunks = zip(self.block_chunks[:-1], self.block_chunks[1:])
        for i, (start, end) in enumerate(chunks):
            msg += f'x{i+1}: blocks[{start}:{end}]\n'
        msg += '\n'
        msg += f'Out shapes (x4, x3, x2, x1, x0): {self._out_shapes}\n'
        msg += str(self._global_params)
        print(msg)


url_map = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',
}

efficientnet_encoders = {
    'efficientnet-b0': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b0'],
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (320, 112, 40, 24, 32),
        'params': {
            'width_coeff': 1.0,
            'depth_coeff': 1.0,
            'image_size': 224,
            'dropout_rate': 0.2,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 2, 5, 11, 16]
        },
    },

    'efficientnet-b1': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b1'],
                'input_space': 'RGB',
                'input_size': [3, 240, 240],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (320, 112, 40, 24, 32),
        'params': {
            'width_coeff': 1.0,
            'depth_coeff': 1.1,
            'image_size': 240,
            'dropout_rate': 0.2,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 3, 8, 16, 23]
        },
    },

    'efficientnet-b2': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b2'],
                'input_space': 'RGB',
                'input_size': [3, 260, 260],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (352, 120, 48, 24, 32),
        'params': {
            'width_coeff': 1.1,
            'depth_coeff': 1.2,
            'image_size': 260,
            'dropout_rate': 0.3,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 3, 8, 16, 23]
        },
    },

    'efficientnet-b3': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b3'],
                'input_space': 'RGB',
                'input_size': [3, 300, 300],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (384, 136, 48, 32, 40),
        'params': {
            'width_coeff': 1.2,
            'depth_coeff': 1.4,
            'image_size': 300,
            'dropout_rate': 0.3,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 3, 8, 18, 26]
        },
    },

    'efficientnet-b4': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b4'],
                'input_space': 'RGB',
                'input_size': [3, 380, 380],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (448, 160, 56, 32, 48),
        'params': {
            'width_coeff': 1.4,
            'depth_coeff': 1.8,
            'image_size': 380,
            'dropout_rate': 0.4,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 5, 10, 22, 32]
        },
    },

    'efficientnet-b5': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b5'],
                'input_space': 'RGB',
                'input_size': [3, 456, 456],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (512, 176, 64, 40, 48),
        'params': {
            'width_coeff': 1.6,
            'depth_coeff': 2.2,
            'image_size': 456,
            'dropout_rate': 0.4,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 6, 13, 27, 39]
        },
    },

    'efficientnet-b6': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b6'],
                'input_space': 'RGB',
                'input_size': [3, 528, 528],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (576, 200, 72, 40, 56),
        'params': {
            'width_coeff': 1.8,
            'depth_coeff': 2.6,
            'image_size': 528,
            'dropout_rate': 0.5,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 7, 15, 31, 45]
        },
    },

    'efficientnet-b7': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': url_map['efficientnet-b7'],
                'input_space': 'RGB',
                'input_size': [3, 600, 600],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (640, 224, 80, 48, 64),
        'params': {
            'width_coeff': 2.0,
            'depth_coeff': 3.1,
            'image_size': 600,
            'dropout_rate': 0.5,
            'drop_connect_rate': 0.2,
            'block_chunks': [0, 9, 18, 38, 55]
        },
    },
}


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)

class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def get_global_params(width_coeff, depth_coeff, image_size, dropout_rate, drop_connect_rate):
    """ Map EfficientNet model name to parameter coefficients. """

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=1000,
        width_coefficient=width_coeff,
        depth_coefficient=depth_coeff,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )
    return global_params



