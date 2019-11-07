import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.weights import transfer_weights


class DeepLabV3(nn.Module):

    def __init__(
        self,
        encoder_name='deeplabv3_resnet101',
        pretrained=False,
        classes=1,
        in_channels=3,
    ):
        super().__init__()
        self.in_channels = in_channels if isinstance(in_channels, int) else len(in_channels)
        self.rgb_channels = in_channels if isinstance(in_channels, str) else 'rgb'

        model = torch.hub.load('pytorch/vision', encoder_name, pretrained=pretrained)
        self.encoder = model.backbone

        # change input channels
        if self.in_channels != 3:
            new_conv = nn.Conv2d(self.in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)
            transfer_weights(self.encoder.conv1, new_conv, method='cycle')
            self.encoder.conv1 = new_conv

        # change output channels
        self.decoder = Decoder(model.classifier, classes)
        self.name = 'deeplabv3-{}'.format(encoder_name)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.encoder(x)  # contract: features is a dict of tensors

        x = features["out"]
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class Decoder(nn.Module):

    def __init__(self, classifier, classes):
        super().__init__()
        self.classifier = classifier
        self.classifier[4] = nn.Conv2d(256, classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.classifier(x)
        return x