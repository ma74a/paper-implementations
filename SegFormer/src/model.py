import torch
import torch.nn as nn
from timm.models.layers import DropPath

from typing import Tuple

class OverlapPatchEmbedding(nn.Module):
    """
    Converts an image or feature map into a sequence of tokens that the Transformer can process.
    Overlapping Patch Merging layer used at the start of each MiT stage.
    Downsamples spatial resolution while increasing channel capacity.
    The output of this is sequence of tokens not 2D feature map
    This will be reused for all four encoder stages, but with different configs
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
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x shape -> [B, C, H, W]
        # apply conv layer shape -> [B, embed_dim, new_H, new_W]
        x = self.conv(x)
        B, C, H, W = x.shape
        
        # then flatten the x across the new_H, new_W dims
        # the shape [B, embed_dim, new_H*new_W]
        x = x.flatten(start_dim=2)
        
        # Each token represents one spatial location, described by a D-dimensional feature vector.
        # then traspose embed_dim with flatten_dim which is N (num_tokens)
        # shape -> [B, N, D]
        x = x.transpose(1, 2)
        
        return x, H, W
        
        
class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        attn_dropout: float=0.0,
        proj_dropout: float=0.0
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        
        # Queries are computed from every input token.
        self.q = nn.Linear(embed_dim, embed_dim)
        # key and value togther but we'll split them later
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        
        # Spatial reduction via Conv2d with kernel_size = stride = sr_ratio
        # Spatially reduce the feature map before computing K and V.
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio
            )
            self.norm = nn.LayerNorm(embed_dim)
            
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        # x shape -> [B, N, D]
        B, N, D = x.shape
        if N != H * W:
            raise ValueError("Token count must equal height * width.")
        
        # compute query
        # convert form [B, N, D] -> [B, N, num_heads, head_dim] then 
        # become [B, hum_heads, N, head_dim]
        # x [B, N, D]
        # q [D, D]
        # q [B, N, D]
        q = self.q(x)
        # we can use transpose instead of permute transpose(1, 2)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply Spatial Reduction for Key and Value if sr_ratio > 1
        # The spatial reduction reduces the number of K/V tokens, not their embedding dimension.
        # It controls how much the spatial dimensions are reduced
        # before generating the Keys and Values for attention.
        if self.sr_ratio > 1:
            # convert x back to be 2D spatial map
            # x [B, N, D] then -> [B, D, H, W] D also consider C no_channels
            x_2d = x.permute(0, 2, 1).reshape(B, D, H, W)
            # then reduece spatial dimenions to [B, D, H/sr, W/sr]
            x_reduced = self.sr(x_2d)
            # then conver this into seq of tokens as before
            # convert it from [B, D, H/sr, W/sr] -> [B, D, H/sr*W/sr] -> [B, N_reduced, D]
            x_reduced = x_reduced.reshape(B, D, -1).permute(0, 2, 1)
            # apply norm
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            
        # each one has [B, num_heads, N_reduced, head_dim]
        k, v = kv[0], kv[1]
        
        # Attention score matrix: [B, num_heads, N, N_reduced]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Aggregate Values: [B, num_heads, N, head_dim] -> [B, N, D]
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x shape -> [B, N, D]
        return x
            
            
class MixFFN(nn.Module):
    """
    The key idea is that a standard Transformer feed-forward network processes each token independently.
    SegFormer adds a depthwise convolution between its two linear projections, 
    allowing the network to incorporate local spatial information.
    3x3 Depthwise Convolution between two Linear layers
    xout = MLP(GELU(Conv3×3 (MLP(xin )))) + xin 
    """
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: int,
        dropout_rate: float=0.0
    ) -> None:
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        # x shape -> [B, N, D]
        # first apply fc1
        # before [B, N, D] after [B, N, new_D]
        x = self.fc1(x)
        B, N, new_D = x.shape
        
        # reshape it and convert it into 2D to pass it to conv layer
        # first to be [B, new_D, N] then [B, new_D, H, W]
        x = x.transpose(1, 2).reshape(B, new_D, H, W)
        # Mix local spatial information without mixing channels.
        x = self.dwconv(x)
        # reshape it again to be [B, N, new_D]
        x = x.flatten(2).transpose(1, 2)
        
        x = self.act(x)
        x = self.dropout(x)
        
        # shape before [B, N, new_D]
        # after [B, N, D]
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x
    
    
class MiTTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attn_dropout=dropout,
            proj_dropout=dropout
        )
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.mlp = MixFFN(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout
        )
        self.drop_path = DropPath(drop_path)
        
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        # Sublayer 1: pre-norm attention + residual connection.
        x = x + self.drop_path(self.attn(self.norm_1(x), H, W))
        # Sublayer 2: pre-norm Mix-FFN + residual connection.
        x = x + self.drop_path(self.mlp(self.norm_2(x), H, W))
        
        return x

