import torch
import torch.nn as nn

class Patchcreation(nn.Module):
    def __init__(
        self,
        input_channel: int,
        patch_size: int,
        embedding_dim: int
    ) -> None:
        self.patch_size = patch_size

        # use conv2d to split the images into patches
        self.patching_conv = nn.Conv2d(
            in_channels=input_channel,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        # then flatte the pathcing conv 
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        # x.shape -> [B, C, H, W]
        # last one -> W, H = W
        img_dim = x.shape[-1]

        # check if img_dim is divideable by patch size or not
        assert img_dim % self.patch_size == 0, \
        f"Given image dimension {img_dim} is not divisible by patch size {self.patch_size}"

        # check if it's one image, add a dim for batch at first
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # input x -> [B, C, H, W]
        # output x => [B, D, H/P, W/P] , D is embed dim
        # H, W will divided by patch size and became 14, 14 if patch size is 16
        x = self.patching_conv(x)

        # input x -> [B, D, H/P, W/P]
        # output x -> [B, D, N]
        # N is H/P * W/P which 14*14
        x = self.flatten(x)

        # change the order to make the N first
        # [B, N, D]
        x = x.permute(0, 2, 1)

        # Batch × Number_of_tokens × Embedding_dimension
        return x
