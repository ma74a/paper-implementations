import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_arch = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            
            # conv4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            
            # conv5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(in_features=4096, out_features=num_classes)
        )
    
    def forward(self, x):
        out = self.conv_arch(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out