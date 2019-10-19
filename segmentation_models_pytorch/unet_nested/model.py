from .decoder import NestedUNetDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class UnetNested(EncoderDecoder):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)
        attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=1,
            activation='sigmoid',
            deep_supervision=False,
            in_channels=3,
            dropout=0,
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels
        )

        decoder = NestedUNetDecoder(
            encoder_channels=encoder.out_shapes,
            out_ch=classes,
            deep_supervision=deep_supervision,
            dropout=dropout,
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-nested-{}'.format(encoder_name)
