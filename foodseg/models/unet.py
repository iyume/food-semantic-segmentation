import torch
from torch import nn
from typing import Optional


class DoubleConv(nn.Sequential):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        if mid_channels is None:
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
    def __init__(self, inc: int, outc: int, midc: Optional[int] = None):
        assert inc < outc
        super().__init__(nn.MaxPool2d(2), DoubleConv(inc, outc, midc))


class Up(nn.Sequential):
    def __init__(self, inc: int, outc: int, midc: Optional[int] = None):
        assert inc > outc
        if midc is None:
            midc = inc
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear"), DoubleConv(inc, outc, midc)
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.outc = DoubleConv(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.outc(x)
        return x
