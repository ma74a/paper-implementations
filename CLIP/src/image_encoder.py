import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        image_size: int=224,
        patch_size: int=16,
        embedding_dim: int=256
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        
        if image_size % patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size"
            )
        
        # num_pathces = (224 // 16) ** 2 => 14 * 14 = 196
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patching_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape -> [B, C, H, W]
        # after conv -> [B, D, H/patch_size, W/patch_size] # D is embed_dim
        x = self.patching_conv(x)
        
        # flatten last two dim
        # before -> [B, D, H/P, W/P] -> [B, D, H/P*W/P]
        # after -> [B, D, N] -> N num of pathces 14*14
        x = x.flatten(start_dim=2)
        
        # we need to transpose last two dim
        # from [B, D, N] to [B, N, D]
        x = x.transpose(1, 2)
        
        return x
    
    


            