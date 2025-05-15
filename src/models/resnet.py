"ResNet"

from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
import torch

from .model import Model


class Resnet(Model):
    """Encoder-decoder"""

    def __init__(self, **args):
        super().__init__(**args)

        self.encoder = (
            resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            if args.get("pretrained_encoder")
            else resnet34()
        )

        self.skip_dimensions = [[-1, 4], [64, 2], [128, 2], [256, 2]]

        self.decoders = torch.nn.ModuleList(
            [
                task.decoder(
                    input_channels=512,
                    output_channels=task.channels,
                    skip_dimensions=self.skip_dimensions,
                )
                for task in self.tasks
            ]
        )

        self.encoder.apply(self._init_weights)

    def forward(self, x):
        "Forward step"

        # remove inputs dimension
        x = x[:, 0, ...]

        # x = (b, c, h, w)

        partial_maps = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        enc_skip_1 = x
        partial_maps.append(enc_skip_1)

        x = self.encoder.layer2(x)
        enc_skip_2 = x
        partial_maps.append(enc_skip_2)

        x = self.encoder.layer3(x)
        enc_skip_3 = x
        partial_maps.append(enc_skip_3)

        x = self.encoder.layer4(x)

        x = [decoder(x, partial_maps) for decoder in self.decoders]

        # Reassemble to (batch, outputs, channels, h, w)
        x = torch.swapaxes(torch.stack(x), 0, 1)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.ones_(m.bias)
