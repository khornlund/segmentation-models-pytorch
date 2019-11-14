import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings

from segmentation_models_pytorch.common.weights import select_rgb_weights


class ResNetEncoder(ResNet):

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels if isinstance(in_channels, int) else len(in_channels)
        self.rgb_channels = in_channels if isinstance(in_channels, str) else 'rgb'
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        if self.in_channels != 3:
            state_dict = self.modify_in_channel_weights(state_dict, self.rgb_channels)
        super().load_state_dict(state_dict, **kwargs)

    def modify_in_channel_weights(self, state_dict, rgb_channels):
        self.conv1 = nn.Conv2d(self.in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)
        pretrained = state_dict['conv1.weight']
        cycled_weights = select_rgb_weights(pretrained, rgb_channels)
        state_dict['conv1.weight'] = cycled_weights
        return state_dict


resnet_encoders = {
    'resnet18': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet18'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },

    'resnet34': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet34'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },

    'resnet50': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet50'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },

    'resnet101': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet101'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
    },

    'resnet152': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet152'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
    },

    'resnext50_32x4d': {
        'encoder': ResNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
            'groups': 32,
            'width_per_group': 4
        },
    },

    'resnext101_32x8d': {
        'encoder': ResNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            },
            'instagram': {
                'url': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            },
            'ssl_yfcc100m': {
                'url': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            },
            'swsl_yfcc100m': {
                'url': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            },
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 8
        },
    },

    'resnext101_32x16d': {
        'encoder': ResNetEncoder,
        'pretrained_settings': {
            'instagram':{
                'url': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 16
        },
    },

    'resnext101_32x32d': {
        'encoder': ResNetEncoder,
        'pretrained_settings': {
            'instagram': {
                'url': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 32
        },
    },

    'resnext101_32x48d': {
        'encoder': ResNetEncoder,
        'pretrained_settings': {
            'instagram': {
                'url': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 48
        },
    },
}
