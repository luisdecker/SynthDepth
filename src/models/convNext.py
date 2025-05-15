"ConvNext"

from torchvision.models import (
    convnext_tiny,
    convnext_base,
    convnext_large,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)
from torchvision.transforms._presets import SemanticSegmentation

from torch import nn
import torch

from .model import Model


class ConvNext(Model):
    """Encoder-decoder"""

    def __init__(self, **args):
        super().__init__(**args)
        encoders = {
            "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights),
            "base": (convnext_base, ConvNeXt_Base_Weights),
            "large": (convnext_large, ConvNeXt_Large_Weights),
        }
        if args.get("encoder_name"):
            encoder_name = args.get("encoder_name")
        else:
            encoder_name = "tiny"

        encoder, weights = encoders[encoder_name]

        self.encoder = (
            encoder(weights=weights.IMAGENET1K_V1)
            if args.get("pretrained_encoder")
            else encoder()
        ).features

        self.transforms = SemanticSegmentation(resize_size=None)
        # if args.get("pretrained_encoder"):
        # self.transforms = weights.IMAGENET1K_V1.transforms()

        skip_dims = {
            "tiny": [[-1, 4], [96, 2], [192, 2], [384, 2]],
            "base": [[-1, 4], [128, 2], [256, 2], [512, 2]],
            "large": [[-1, 4], [192, 2], [384, 2], [768, 2]],
        }

        self.skip_dimensions = skip_dims[encoder_name]

        input_channels = {"tiny": 768, "base": 1024, "large": 1536}
        self.decoders = torch.nn.ModuleList(
            [
                task.decoder(
                    input_channels=input_channels[encoder_name],
                    output_channels=task.channels,
                    skip_dimensions=self.skip_dimensions,
                    **task.decoder_args
                )
                for task in self.tasks
            ]
        )

        # self.apply(self._init_weights)

    def forward(self, x):
        "Forward step"

        # remove inputs dimension
        x = x[:, 0, ...]

        # x = (b, c, h, w)

        # Apply pretrained transforms
        if self.transforms:
            x = self.transforms(x)

        partial_maps = []  # Channels are for base size
        x = self.encoder[0](x)  # -> x = (b, 128, h/4, w/4) ( ^*4 )
        x = self.encoder[1](x)  # -> x = (b, 128, h/4, w/4)
        enc_skip_1 = x
        partial_maps.append(enc_skip_1)

        x = self.encoder[2](x)  # -> x = (b, 256, h/8, w/8) ( ^*2 )
        x = self.encoder[3](x)  # -> x = (b, 256, h/8, w/8)
        enc_skip_2 = x
        partial_maps.append(enc_skip_2)

        x = self.encoder[4](x)  # -> x = (b, 512, h/16, w/16) ( ^*2 )
        x = self.encoder[5](x)  # -> x = (b, 512, h/16, w/16)
        enc_skip_3 = x
        partial_maps.append(enc_skip_3)

        x = self.encoder[6](x)  # -> x = (b, 1024, h/32, w/32) ( ^*2 )
        x = self.encoder[7](x)

        x = [decoder(x, partial_maps) for decoder in self.decoders]

        # Expand to largest feature size
        x = self.expand_shape(x)

        # Reassemble to (batch, tasks, channels, h, w)
        x = torch.swapaxes(torch.stack(x), 0, 1)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.ones_(m.bias)
