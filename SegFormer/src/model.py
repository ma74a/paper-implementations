import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from typing import Tuple, Sequence, List

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
        x = self.layer_norm(x)
        
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
        
        # shape -> [B, N, D]
        return x
    
    

class MixVisionTransformer(nn.Module):
    """
    MiT encoder: 4 stages of [OverlapPatchEmbedding -> L x MiTTransformerBlock -> LayerNorm].
    Returns the four stage outputs as 2D feature maps at strides 4, 8, 16, 32.
    each stage downsamples the spatial resolution while increasing the number of channels.
    Multiple Transformer blocks: each stage repeatedly refines its token representations.
    Multi-scale outputs: the encoder saves all four feature maps for the decoder
    """
    def __init__(
        self,
        in_channels: int=3,
        embed_dims: Sequence[int]=(32, 64, 160, 256),
        depths: Sequence[int]=(2, 2, 2, 2),
        num_heads: Sequence[int]=(1, 2, 5, 8),
        sr_ratios: Sequence[int]=(8, 4, 2, 1),
        mlp_ratios: Sequence[int]=(4, 4, 4, 4),
        patch_kernels: Sequence[int]=(7, 3, 3, 3),
        patch_strides: Sequence[int]=(4, 2, 2, 2),
        patch_paddings: Sequence[int]=(3, 1, 1, 1),
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_stages = len(depths)
        
        # four patch embedding for each 4 stages
        self.patch_embedding_1 = OverlapPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=patch_kernels[0],
            stride=patch_strides[0],
            padding=patch_paddings[0]
        )
        self.patch_embedding_2 = OverlapPatchEmbedding(
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=patch_kernels[1],
            stride=patch_strides[1],
            padding=patch_paddings[1]
        )
        self.patch_embedding_3 = OverlapPatchEmbedding(
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=patch_kernels[2],
            stride=patch_strides[2],
            padding=patch_paddings[2]
        )
        self.patch_embedding_4 = OverlapPatchEmbedding(
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=patch_kernels[3],
            stride=patch_strides[3],
            padding=patch_paddings[3]
        )
        
        # Stochastic depth rate grows linearly from 0 to drop_path_rate across ALL blocks.
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist() # stochastic depth decay rule
        cur = 0
        self.block_1 = nn.ModuleList([
            MiTTransformerBlock(
                embed_dim=embed_dims[0],
                num_heads=num_heads[0],
                sr_ratio=sr_ratios[0],
                mlp_ratio=mlp_ratios[0],
                dropout=dropout,
                drop_path=dpr[cur+i]
            )
            for i in range(depths[0])
        ])
        self.norm_1 = nn.LayerNorm(embed_dims[0])
        
        cur += depths[0]
        self.block_2 = nn.ModuleList([
            MiTTransformerBlock(
            embed_dim=embed_dims[1],
            num_heads=num_heads[1],
            sr_ratio=sr_ratios[1],
            mlp_ratio=mlp_ratios[1],
            dropout=dropout,
            drop_path=dpr[cur+i]
        )
            for i in range(depths[1])
        ])
        self.norm_2 = nn.LayerNorm(embed_dims[1])
        
        cur += depths[1]
        self.block_3 = nn.ModuleList([
            MiTTransformerBlock(
            embed_dim=embed_dims[2],
            num_heads=num_heads[2],
            sr_ratio=sr_ratios[2],
            mlp_ratio=mlp_ratios[2],
            dropout=dropout,
            drop_path=dpr[cur+i]
        )
            for i in range(depths[2])
        ])
        self.norm_3 = nn.LayerNorm(embed_dims[2])
        
        cur += depths[2]
        self.block_4 = nn.ModuleList([
            MiTTransformerBlock(
            embed_dim=embed_dims[3],
            num_heads=num_heads[3],
            sr_ratio=sr_ratios[3],
            mlp_ratio=mlp_ratios[3],
            dropout=dropout,
            drop_path=dpr[cur+i]
        )
            for i in range(depths[3])
        ])
        self.norm_4 = nn.LayerNorm(embed_dims[3])
        
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x shape -> [B, C, H, W]
        B, *_ = x.shape
        outs = []
        
        # stage1 
        x, H, W = self.patch_embedding_1(x)
        for i, blk in enumerate(self.block_1):
            x = blk(x, H, W)
        # shape -> [B, N, D]
        x = self.norm_1(x)
        # we want the shape to be [B, C, H, W]
        # tokens back to a 2D map: [B, N, D] -> [B, D, H, W].
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # stage2 
        x, H, W = self.patch_embedding_2(x)
        for i, blk in enumerate(self.block_2):
            x = blk(x, H, W)
        # shape -> [B, N, D]
        x = self.norm_2(x)
        # we want the shape to be [B, C, H, W]
        # tokens back to a 2D map: [B, N, D] -> [B, D, H, W].
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        
        # stage3 
        x, H, W = self.patch_embedding_3(x)
        for i, blk in enumerate(self.block_3):
            x = blk(x, H, W)
        # shape -> [B, N, D]
        x = self.norm_3(x)
        # we want the shape to be [B, C, H, W]
        # tokens back to a 2D map: [B, N, D] -> [B, D, H, W].
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        
        # stage4
        x, H, W = self.patch_embedding_4(x)
        for i, blk in enumerate(self.block_4):
            x = blk(x, H, W)
        # shape -> [B, N, D]
        x = self.norm_4(x)
        # we want the shape to be [B, C, H, W]
        # tokens back to a 2D map: [B, N, D] -> [B, D, H, W].
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        return outs
        
        
class MLP(nn.Module):
    """Linear projection applied independently at every spatial position."""
    def __init__(
        self,
        input_dim: int,
        embed_dim: int
    ) -> None:
        super().__init__()
        self.linear_proj = nn.Linear(in_features=input_dim, out_features=embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape -> [B, C, H, W]
        # convert the shape -> [B, C, H*W]
        # then transpose it -> [B, H*W, C]
        x = x.flatten(start_dim=2).transpose(1, 2)
        
        # x before [B, N, D]
        # x after [B, N, new_D]
        x = self.linear_proj(x)
        
        return x
        
class SegFormerHead(nn.Module):
    """
    SegFormer All-MLP decode,
    Projects four encoder stages into a shared embedding dimension,
    resizes them to the highest-resolution stage, concatenates and fuses
    the features, then predicts per-pixel class logits.
    """
    def __init__(
        self,
        in_channels: List[int]=[32, 64, 160, 256],
        embed_dim: int=256,
        num_classes: int=19,
        dropout_rate: float=0.1
    ) -> None:
        super().__init__()
        # Stage-specific MLP projections.
        self.linear_c1 = MLP(in_channels[0], embed_dim)
        self.linear_c2 = MLP(in_channels[1], embed_dim)
        self.linear_c3 = MLP(in_channels[2], embed_dim)
        self.linear_c4 = MLP(in_channels[3], embed_dim)
        
        # Fuse concatenated multi-scale features.
        # after we contcate the four linears across channel dim
        # we get [B, 1024, H, W] so want C_dim to be equal embed_dim
        # we use kernel_size=1 to not change the H, W dim
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim*4,
                out_channels=embed_dim,
                kernel_size=1
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(dropout_rate)
        # same here but for predict classes
        self.linear_pred = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=num_classes,
            kernel_size=1
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features are what we get from the encoder
        # list of [F1, F2, F3, F4]
        # each one of shape -> [B, D, H_red, W_red]
        f1, f2, f3, f4 = features
        
        # Remember the highest-resolution spatial dimensions.
        # f1 has the highest H, W so we'll use it to make other features similar it
        target_size = f1.shape[2:]
        # get the batch
        n = f1.shape[0]
        
        # all this will give the same embed_dim but different H, W
        # [B, N, embed_dim]
        c1 = self.linear_c1(f1)
        c2 = self.linear_c2(f2)
        c3 = self.linear_c3(f3)
        c4 = self.linear_c4(f4)
        
        # then convert each one from [B, N, embed_dim] to [B, D, H, W]
        c1 = c1.permute(0, 2, 1).reshape(n, -1, features[0].shape[2], features[0].shape[3])
        c2 = c2.permute(0, 2, 1).reshape(n, -1, features[1].shape[2], features[1].shape[3])
        c3 = c3.permute(0, 2, 1).reshape(n, -1, features[2].shape[2], features[2].shape[3])
        c4 = c4.permute(0, 2, 1).reshape(n, -1, features[3].shape[2], features[3].shape[3])
        
        # Resize lower-resolution stages to the resolution of C1.
        # make all H, W similar to c1 
        c2 = F.interpolate(
            c2,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        c3 = F.interpolate(
            c3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        c4 = F.interpolate(
            c4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        
        # then concatenate all of them 
        # across C dim started with c4
        # x shape [B, C*4, highest_H, highest_W]
        x = torch.cat([c4, c3, c2, c1], dim=1)
        
        # Fuse features, apply dropout, and predict segmentation logits.
        # convert x shape to [B, embed_dim, highest_H, highest_W]
        x = self.linear_fuse(x)
        x = self.dropout(x)
        # now x shape -> [B, num_classes, highest_H, highest_W]
        x = self.linear_pred(x)

        return x