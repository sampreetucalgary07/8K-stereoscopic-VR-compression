## Code reference from : https://github.com/explainingai-code/VAE-Pytorch/tree/main/model
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(__file__))


######################## DOWNBLOCK ########################
class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        down_sample,
        num_layers,
        norm_channels,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = (
            nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            if self.down_sample
            else nn.Identity()
        )

    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        # Downsample
        out = self.down_sample_conv(out)
        return out


######################## UPBLOCK ########################


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        up_sample,
        num_layers,
        attn,
        norm_channels,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = (
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
            if self.up_sample
            else nn.Identity()
        )

    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)

        # Concat with Downblock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
        return out


######################## CONDITIONAL BLOCKS ########################
# conditional conv
class ConditionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConditionalConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.cond_scale = nn.Linear(1, out_channels)
        self.cond_shift = nn.Linear(1, out_channels)

    def forward(self, x, beta):
        x = self.conv(x)
        scale = self.cond_scale(beta).unsqueeze(2).unsqueeze(3)
        shift = self.cond_shift(beta).unsqueeze(2).unsqueeze(3)
        return x * scale + shift


# conditonal resnet layer
class ConditionalResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_channels):
        super(ConditionalResNetLayer, self).__init__()
        self.gn = nn.GroupNorm(norm_channels, in_channels)
        self.silu = nn.SiLU()
        self.conv = ConditionalConv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, beta):
        x = self.gn(x)
        x = self.silu(x)
        x = self.conv(x, beta)
        return x
