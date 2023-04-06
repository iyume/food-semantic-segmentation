import torch
from torch import nn


class DoubleConv(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class Down(nn.Sequential):
    def __init__(self, inc, outc):
        super().__init__(nn.MaxPool2d(2), DoubleConv(inc, outc))


class Up(nn.Sequential):
    def __init__(self, inc, outc):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear"), DoubleConv(inc, outc)
        )


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.outc = DoubleConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.outc(x)
        return x
