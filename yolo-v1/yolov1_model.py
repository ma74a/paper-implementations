import torch
import torch.nn as nn

S=7 # no of grid cells
B=2 # no of bbox for each cell predicts
C=20 # no of classes

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            maxpool_flag=False
    ):
        super(ConvBlock, self).__init__()
        self.maxpool_flag = maxpool_flag
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        if self.maxpool_flag:
            x = self.max_pool(x)

        return x


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # in_channels, out_channels, kernel_size, stride, padding, max_pool
        self.stage1 = ConvBlock(3, 64, 7, 2, 3, maxpool_flag=True)
        self.stage2 = ConvBlock(64, 192, 3, 1, 1, maxpool_flag=True)

        self.stage3 = nn.ModuleList([
                ConvBlock(192, 128, 1, 1, 0),
                ConvBlock(128, 256, 3, 1, 1),
                ConvBlock(256, 256, 1, 1, 0),
                ConvBlock(256, 512, 3, 1, 1, maxpool_flag=True)
            ])

        self.stage4 = nn.ModuleList([])
        for _ in range(4):
            self.stage4.append(ConvBlock(512, 256, 1, 1, 0))
            self.stage4.append(ConvBlock(256, 512, 3, 1, 1))
        self.stage4.append(ConvBlock(512, 512, 1, 1, 0))
        self.stage4.append(ConvBlock(512, 1024, 3, 1, 1, maxpool_flag=True))

        self.stage5 = nn.ModuleList([])
        for _ in range(2):
            self.stage5.append(ConvBlock(1024, 512, 1, 1, 0))
            self.stage5.append(ConvBlock(512, 1024, 3, 1, 1))
        self.stage5.append(ConvBlock(1024, 1024, 3, 1, 1))
        self.stage5.append(ConvBlock(1024, 1024, 3, 2, 1)) # stride=2

        self.stage6 = nn.ModuleList([
                ConvBlock(1024, 1024, 3, 1, 1),
                ConvBlock(1024, 1024, 3, 1, 1)
            ])

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        
        for layer in self.stage3:
            x = layer(x)
        
        for layer in self.stage4:
            x = layer(x)

        for layer in self.stage5:
            x = layer(x)

        for layer in self.stage6:
            x = layer(x)

        return x


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(in_features=1024*7*7, out_features=4096)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=(B*5+C)*S*S)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.conv_backbone = Backbone()
        self.fc = FullyConnected()

    def forward(self, x):
        x = self.conv_backbone(x)
        print(f"before flatten: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        print(f"after flatten: {x.size()}")
        x = self.fc(x)
        return x

model = YOLOv1()
x = torch.randn(1, 3, 448, 448)
out = model(x)
print(out.shape)
