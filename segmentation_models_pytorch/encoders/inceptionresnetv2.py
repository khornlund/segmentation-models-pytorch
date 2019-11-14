import torch.nn as nn
from pretrainedmodels.models.inceptionresnetv2 import InceptionResNetV2
from pretrainedmodels.models.inceptionresnetv2 import pretrained_settings

from segmentation_models_pytorch.common.weights import select_rgb_weights


class InceptionResNetV2Encoder(InceptionResNetV2):

    def __init__(self, in_channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels if isinstance(in_channels, int) else len(in_channels)
        self.rgb_channels = in_channels if isinstance(in_channels, str) else 'rgb'
        self.pretrained = False

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.avgpool_1a
        del self.last_linear

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x0 = x

        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x1 = x

        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x2 = x

        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x3 = x

        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x4 = x

        features = [x4, x3, x2, x1, x0]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        if self.in_channels != 3:
            state_dict = self.modify_in_channel_weights(state_dict, self.rgb_channels)
        super().load_state_dict(state_dict, **kwargs)

    def modify_in_channel_weights(self, state_dict, rgb_channels):
        self.conv2d_1a.conv = nn.Conv2d(self.in_channels, 32, (3, 3), (2, 2), (1, 1), bias=False)
        pretrained = state_dict['conv2d_1a.conv.weight']
        cycled_weights = select_rgb_weights(pretrained, rgb_channels)
        state_dict['conv2d_1a.conv.weight'] = cycled_weights
        return state_dict


inception_encoders = {
    'inceptionresnetv2': {
        'encoder': InceptionResNetV2Encoder,
        'pretrained_settings': {
            'ens_adv': {
                'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ens_adv_inception_resnet_v2-2592a550.pth',
                'input_space': 'RGB',
                'input_size': [3, 299, 299],
                'input_range': [0, 1],
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'num_classes': 1000
            },
            **pretrained_settings['inceptionresnetv2'],
        },
        'out_shapes': (1536, 1088, 320, 192, 64),
        'params': {
            'num_classes': 1000,
        }

    }
}
