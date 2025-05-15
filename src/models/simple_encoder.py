"Basic encoder-decoder"

from .model import Model

import torch
import torch.nn as nn
from .task import DenseRegression


class Encoder(nn.Module):
    """Encoder"""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.partial_maps = []

    def forward(self, x):

        # Swap axis
        # apply convs
        x = self.conv1(x)
        partial_map_1 = x

        x = self.conv2(x)
        partial_map_2 = x

        x = self.conv3(x)

        return x, [partial_map_1, partial_map_2]


class SimpleEncoder(Model):
    """Encoder-decoder"""

    def __init__(self, **args):
        super().__init__(**args)

        self.encoder = Encoder()

        for task in self.tasks:
            task.decoder = task.decoder(
                input_channels=128, output_channels=task.channels
            )

    def forward(self, x):
        "Forward step"

        # remove inputs dimension
        x = x[:, 0, ...]

        x, partial_maps = self.encoder(x)

        x = [task.decoder(x, partial_maps) for task in self.tasks]

        # Reassemble to (batch, outputs, channels, h, w)
        x = torch.swapaxes(torch.stack(x), 0, 1)
        return x
