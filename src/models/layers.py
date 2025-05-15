import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleConv(nn.Module):
    "Upsample followed by convolution"

    def __init__(self, in_channels, out_channels, **args) -> None:
        super().__init__()

        self.pixel_shuffle = args.get("pixel_shuffle", False)

        kernel_size = args.get("kernel_size", 3 if not self.pixel_shuffle else 7)
        stride = args.get("stride", 1)
        padding = args.get("padding", "same")

        self.batchnorm = args.get("batchnorm", False)
        if self.batchnorm:
            self.normalization = nn.BatchNorm2d(out_channels)

        self.activation = args.get("activation", nn.Mish)()

        print("Using pixel shuffle: ", self.pixel_shuffle)
        self.upsample = (
            nn.UpsamplingBilinear2d(scale_factor=2)
            if not self.pixel_shuffle
            else nn.PixelShuffle(upscale_factor=2)
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.pixel_shuffle_conv = (
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 4,
                kernel_size=1,
                padding="same",
            )
            if self.pixel_shuffle
            else None
        )

    def forward(self, x):

        if self.pixel_shuffle:
            x = self.pixel_shuffle_conv(x)
        x = self.upsample(x)
        x = self.conv(x)
        if self.batchnorm:
            x = self.normalization(x)
        x = self.activation(x)

        return x


class PSUpsampleConv(nn.Module):
    "Upsample followed by convolution more suitable for pixel shuffle"

    def __init__(self, in_channels, out_channels, **args) -> None:
        super().__init__()

        self.pixel_shuffle = args.get("pixel_shuffle", False)

        kernel_size = args.get("kernel_size", 3 if not self.pixel_shuffle else 7)
        stride = args.get("stride", 1)
        padding = args.get("padding", "same")

        self.batchnorm = args.get("batchnorm", False)
        if self.batchnorm:
            self.normalization_last = nn.BatchNorm2d(out_channels)
            self.normalization_first = nn.BatchNorm2d(in_channels)

        self.activation = args.get("activation", nn.Mish)()

        print("Using pixel shuffle: ", self.pixel_shuffle)
        self.upsample = (
            nn.UpsamplingBilinear2d(scale_factor=2)
            if not self.pixel_shuffle
            else nn.PixelShuffle(upscale_factor=2)
        )
        self.pre_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding="same",
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.pixel_shuffle_conv = (
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 4,
                kernel_size=3,
                padding="same",
            )
            if self.pixel_shuffle
            else None
        )

    def forward(self, x):

        x = self.pre_conv(x)
        if self.batchnorm:
            x = self.normalization_first(x)
        x = self.activation(x)
        if self.pixel_shuffle:
            x = self.pixel_shuffle_conv(x)
        x = self.upsample(x)
        x = self.conv(x)
        if self.batchnorm:
            x = self.normalization_last(x)
        x = self.activation(x)

        return x


class UpscaleConvNext(nn.Module):
    r"""A bilinear upscale followed by a ConvNext block"""

    def __init__(self, channels):
        super().__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block = ConvNextBlock(channels)

    def forward(self, x):

        x = self.upsample(x)
        x = self.block(x)

        return x


class ConvNextBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


"ASPP layer with different dilation rates"


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                    bias=False,
                )
                for rate in dilation_rates
            ]
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.conv1 = nn.Conv2d(
            out_channels * (len(dilation_rates) + 1),
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        aspp_outs = [block(x) for block in self.aspp_blocks]
        global_avg_pool_out = self.global_avg_pool(x)
        global_avg_pool_out = F.interpolate(
            global_avg_pool_out, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        aspp_outs.append(global_avg_pool_out)
        x = torch.cat(aspp_outs, dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        return self.relu(x)


class DenseASPP(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, dilation_rates):
        super(DenseASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList()

        # 1x1 Conv to project input channels to inter_channels
        self.aspp_blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Densely connected dilated convolutions
        for i, rate in enumerate(dilation_rates):
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        inter_channels + i * inter_channels,
                        inter_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(inter_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Final 1x1 convolution to merge outputs
        self.conv1 = nn.Conv2d(
            inter_channels + len(dilation_rates) * inter_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        aspp_outs = [self.aspp_blocks[0](x)]

        # Apply dense connections
        for i in range(1, len(self.aspp_blocks)):
            concatenated_features = torch.cat(aspp_outs, dim=1)
            aspp_outs.append(self.aspp_blocks[i](concatenated_features))

        # Concatenate all ASPP outputs and pass through final 1x1 conv
        x = torch.cat(aspp_outs, dim=1)
        x = self.conv1(x)
        x = self.bn(x)

        return self.relu(x)
