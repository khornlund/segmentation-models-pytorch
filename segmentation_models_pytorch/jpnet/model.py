from .decoder import JPNDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class JPN(EncoderDecoder):
    """
    https://github.com/wuhuikai/FastFCN/blob/master/encoding/models/base.py
    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=1,
            activation='sigmoid',
            dropout=0.1,
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )
        decoder = JPNDecoder(
            encoder_channels=encoder.out_shapes,
            final_channels=classes,
            dropout=dropout,
        )
        super().__init__(encoder, decoder, activation)
        self.name = 'jpn-{}'.format(encoder_name)
