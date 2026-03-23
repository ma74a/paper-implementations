import torch
import torch.nn as nn

from .unet_parts import DoubleConv, Down, Up

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        # first the down
        self.down1 = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down2 = Down(in_channels=64, out_channels=128)
        self.down3 = Down(in_channels=128, out_channels=256)
        self.down4 = Down(in_channels=256, out_channels=512)

        # the bottleneck
        self.bottleneck = DoubleConv(in_channels=512, out_channels=1024)

        # the UP
        self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64)

        # the out layer
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # bottleneck
        bottle_out = self.bottleneck(x4)

        # decoder
        x = self.up1(bottle_out, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)
    


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=2)
    x = torch.randn(1, 3, 250, 250)  # any size
    y = model(x)
    print(y.shape)