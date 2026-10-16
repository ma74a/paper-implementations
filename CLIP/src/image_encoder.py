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
    
    
class AddClassAndPositionEmbedding(nn.Module):
    def __init__(
        self,
        num_patches: int = 196,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        # create a cls token which will be learnable
        # will be of shape [1, 1, 768] to add to patch embedding
        # CLS = learned representation useful for classification
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )
        
        # add one for cls token
        num_positions = num_patches + 1

        # create learnable positional encoding
        # shape of [1, 197, 768]
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_positions, embedding_dim)
        )
        
    def forward(self, patch_embed: torch.Tensor) -> torch.Tensor:
        # x shape -> [B, N, D]
        batch_size = patch_embed.shape[0]
        
        # expand the cls token to add batch_size
        # from [1, 1, 768] to [batch_size, 1, 768]
        cls_token = self.cls_token.expand(batch_size, -1, -1) # -1 mean to change this dim
        
        # we'll concatenate cls token with patch emb
        # [B, N, D]
        # B = batch
        # N = number of tokens -> dim = 1
        # D = embedding dimension
        # cls_token ->     [8, 1, 768]
        # patch_dim ->     [8, 196, 768]
        # output of this-> [8, 197, 768]
        tokens = torch.concat((cls_token, patch_embed), dim=1)
        
        # shape -> [B, N+1, D]
        out = tokens + self.positional_encoding
        
        return out


            