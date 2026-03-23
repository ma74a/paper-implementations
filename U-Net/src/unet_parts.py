import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # (Conv2d -> batchnorm -> Relu) * 2
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)
    

class Down(nn.Module):
    # max_pooling -> DoubleConv
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.down(x)
        return x
    

class Up(nn.Module):
    # Upsampling -> Concatenate skip -> Doubleconv
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # if there is mismatch dim
        # [b, C, h, w]
        diff_h = skip.size(2) - x.size(2) # difference in height
        diff_w = skip.size(3) - x.size(3) # difference in weidth

        if diff_h != 0 or diff_w != 0: # check if there is mismatch, if there we use padding
            x = F.pad(x, [diff_w//2, diff_w - diff_w//2,
                          diff_h//2, diff_h - diff_h//2])
            
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)