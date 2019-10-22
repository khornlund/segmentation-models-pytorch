
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

from segmentation_models_pytorch.common.weights import select_rgb_weights


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            base_width: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super().__init__()
        width = int(math.floor(planes * (base_width/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, in_channels=3, base_width=26, scale=4, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels if isinstance(in_channels, int) else len(in_channels)
        self.rgb_channels = in_channels if isinstance(in_channels, str) else 'rgb'
        self.inplanes = 64
        self.base_width = base_width
        self.scale = scale
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', base_width = self.base_width, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width = self.base_width, scale=self.scale))

        return nn.Sequential(*layers)

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


res2net_encoders = {
    'res2net50_26w_4s': {
        'encoder': Res2Net,
        'pretrained_settings': {
            'imagenet': {
                'url': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_4s-06e79181.pth'
            },
        },
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': Bottle2neck,
            'layers': [3, 4, 6, 3],
            'base_width': 26,
            'scale': 4
        },
    },
    'res2net50_48w_2s': {
        'encoder': Res2Net,
        'pretrained_settings': {
            'imagenet': {
                'url': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_48w_2s-afed724a.pth'
            },
        },
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': Bottle2neck,
            'layers': [3, 4, 6, 3],
            'base_width': 48,
            'scale': 2
        },
    },
    'res2net50_14w_8s': {
        'encoder': Res2Net,
        'pretrained_settings': {
            'imagenet': {
                'url': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_14w_8s-6527dddc.pth'
            },
        },
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': Bottle2neck,
            'layers': [3, 4, 6, 3],
            'base_width': 14,
            'scale': 8
        },
    },
    'res2net50_26w_6s': {
        'encoder': Res2Net,
        'pretrained_settings': {
            'imagenet': {
                'url': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_6s-19041792.pth'
            },
        },
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': Bottle2neck,
            'layers': [3, 4, 6, 3],
            'base_width': 26,
            'scale': 6
        },
    },
    'res2net50_26w_8s': {
        'encoder': Res2Net,
        'pretrained_settings': {
            'imagenet': {
                'url': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_8s-2c7c9f12.pth'
            },
        },
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': Bottle2neck,
            'layers': [3, 4, 6, 3],
            'base_width': 26,
            'scale': 8
        },
    },
    'res2net101_26w_4s': {
        'encoder': Res2Net,
        'pretrained_settings': {
            'imagenet': {
                # 'url': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net101_26w_4s-02a759a1.pth'
                'url': 'https://u08ica.dm.files.1drv.com/y4mIfD6z9eezA6mfn9DmVvtDWhdU_iexSZFVO1kDzH8KYG8cHpwnP5-J7pIPYA1FAGqJMdhiq2EuiamMk8J_9g03CAWJm9hDT9S9BeJerffwpHQlL_v0qww5X1RW3lDdMimIwZZQQCRyUQsMfGvMJRNYVtY2z8Vfwzf1RwPbZB7pRfw8LwaIhxH6BqIXm8blECt2Cd5yduNv5eRthRYwI6nAw/res2net101_26w_4s-02a759a1.pth?download&psid=1'
            },
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottle2neck,
            'layers': [3, 4, 23, 3],
            'base_width': 26,
            'scale': 4
        },
    },
}
