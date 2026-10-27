import torch
import torch.nn as nn

class OverlapPatchEmbeddings(nn.Module):
    """
    Overlapping Patch Merging layer used at the start of each MiT stage.
    Downsamples spatial resolution while increasing channel capacity.
    The output of this is sequence of tokens not 2D feature map
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor):
        # x shape -> [B, C, H, W]
        # apply conv layer shape -> [B, embed_dim, new_H, new_W]
        x = self.conv(x)
        
        # then flatten the x across the new_H, new_W dims
        # the shape [B, embed_dim, new_H*new_W]
        x = x.flatten(start_dim=2)
        
        # then traspose embed_dim with flatten_dim which is N (num_tokens)
        # shape -> [B, N, D]
        x = x.transpose(1, 2)
        
        return x
        
        
if __name__ == "__main__":
    # x = torch.randn(4, 3, 512, 512)
    # model = OverlapPatchEmbeddings(3, 128, 7, 4, 3)
    # out = model(x)
    # print(out.shape)
    pass