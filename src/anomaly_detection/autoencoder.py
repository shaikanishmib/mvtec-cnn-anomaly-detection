"""Scratch convolutional reconstruction baseline; RGB inputs/outputs in [0,1]."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConvAutoencoder(nn.Module):
    """No skip connections: every reconstruction passes through the bottleneck.

    For depth=3, width=16, latent_channels=8 and 256-square inputs:
    encoder: B,3,256,256 -> B,16,128,128 -> B,32,64,64 -> B,64,32,32;
    bottleneck: B,8,32,32; decoder: B,32,64,64 -> B,16,128,128
    -> B,3,256,256. Bilinear upsampling followed by convolution avoids
    uneven transposed-convolution overlap. The final sigmoid bounds RGB.
    """
    def __init__(self, width=16, depth=3, latent_channels=8):
        super().__init__()
        if depth not in (3, 4) or width < 1 or latent_channels < 1:
            raise ValueError("Expected depth 3/4 and positive channel counts")
        self.divisor = 2 ** depth
        channels = [3] + [width * 2**i for i in range(depth)]
        layers = []
        for before, after in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(before, after, 3, 2, 1), nn.ReLU()])
        self.encoder = nn.Sequential(*layers)
        self.bottleneck = nn.Conv2d(channels[-1], latent_channels, 1)
        layers = []
        before = latent_channels
        for after in list(reversed(channels[1:-1])) + [3]:
            layers.extend([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                           nn.Conv2d(before, after, 3, padding=1),
                           nn.Sigmoid() if after == 3 else nn.ReLU()])
            before = after
        self.decoder = nn.Sequential(*layers)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or any(s % self.divisor for s in images.shape[-2:]):
            raise ValueError("Expected B,3,H,W with spatial dimensions divisible by 2**depth")
        return self.decoder(self.bottleneck(self.encoder(images)))


def reconstruction_loss(reconstructed, target, kind="l1"):
    if reconstructed.shape != target.shape:
        raise ValueError("Reconstruction and target shapes must match")
    if kind == "l1":
        return F.l1_loss(reconstructed, target)
    if kind == "mse":
        return F.mse_loss(reconstructed, target)
    raise ValueError("Expected l1 or mse; combined losses require development justification")


def anomaly_map(images, reconstructed, kind="absolute"):
    if images.shape != reconstructed.shape or images.ndim != 4:
        raise ValueError("Matching B,C,H,W tensors required")
    difference = images.float() - reconstructed.float()
    if kind == "absolute":
        return difference.abs().mean(1)
    if kind == "squared":
        return difference.square().mean(1)
    raise ValueError("Expected absolute or squared")


def image_scores(maps, fraction=.01):
    """Fixed top-1% mean by default; independent of image/pixel thresholds."""
    if maps.ndim != 3 or not 0 < fraction <= 1:
        raise ValueError("Expected B,H,W maps and fraction in (0,1]")
    flat = maps.flatten(1)
    return flat.topk(max(1, math.ceil(flat.shape[1]*fraction)), dim=1).values.mean(1)
