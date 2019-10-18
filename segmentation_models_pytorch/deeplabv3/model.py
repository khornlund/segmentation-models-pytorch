import torch
import torch.nn as nn

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
        model = torch.hub.load('pytorch/vision', encoder_name, pretrained=pretrained)

        self.encoder = model.backbone

        # change input channels
        new_conv = nn.Conv2d(in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)
        transfer_weights(self.encoder.conv1, new_conv, method='cycle')
        self.encoder.conv1 = new_conv

        # change output channels
        self.decoder = Decoder(model.classifier, classes)

        self.name = 'u-{}'.format(encoder_name)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.encoder(x)

        result = OrderedDict()
        x = features["out"]
        x = self.decoder.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        x = features["aux"]
        x = self.decoder.aux_classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["aux"] = x

        return result


class Decoder(nn.Module):

    def __init__(self, classifier, classes):
        super().__init__()
        self.classifier = classifier
        self.aux_classifier = FCNHead(classifier[-1].out_channels, classes)

    def forward(self, x):
        x = self.classifier(x)
        x = self.aux_classifier(x)
        return x


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)