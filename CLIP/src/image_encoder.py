import torch
import torch.nn as nn

import math

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
        # will be of shape [1, 1, 256] to add to patch embedding
        # CLS = learned representation useful for classification
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )
        
        # add one for cls token
        num_positions = num_patches + 1

        # create learnable positional encoding
        # shape of [1, 197, 256]
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_positions, embedding_dim)
        )
        
    def forward(self, patch_embed: torch.Tensor) -> torch.Tensor:
        # x shape -> [B, N, D]
        batch_size = patch_embed.shape[0]
        
        # expand the cls token to add batch_size
        # from [1, 1, 768] to [batch_size, 1, 256]
        cls_token = self.cls_token.expand(batch_size, -1, -1) # -1 mean to change this dim
        
        # we'll concatenate cls token with patch emb
        # [B, N, D]
        # B = batch
        # N = number of tokens -> dim = 1
        # D = embedding dimension
        # cls_token ->     [8, 1, 256]
        # patch_dim ->     [8, 196, 256]
        # output of this-> [8, 197, 256]
        tokens = torch.concat((cls_token, patch_embed), dim=1)
        
        # shape -> [B, N+1, D]
        out = tokens + self.positional_encoding
        
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # check if the embed_dim is divisable by num_heads 
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        
        self.num_heads = num_heads
        
        # embed_dim for each head
        self.d_k = embedding_dim // num_heads
        self.scale = math.sqrt(self.d_k)
        
        self.w_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim) # Query
        self.w_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim) # Key
        self.w_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim) # Value
        self.w_o = nn.Linear(in_features=embedding_dim, out_features=embedding_dim) # Out
        
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, embed_dim = x.shape
        
        # shape -> [B, N, num_heads, d_k]
        # d_k is embed_dim // num_heads
        x = x.view(batch_size, num_patches, self.num_heads, self.d_k)
        
        # we want the shape to be -> [B, num_heads, N, d_k]
        # so we use transpose for that
        x = x.transpose(1, 2)
        
        return x
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        dropout: nn.Dropout
    ) -> torch.Tensor:
        # Q shape -> [B, num_heads, N, d_k]
        # K shape before transpose -> [B, num_heads, N, d_k]
        # we want to transpose K shape to match Q shape to we apply matmul
        # K after transpose -> [B, num_heads, d_k, N]
        # output shape -> [B, num_heads, N, N]
        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # apply softmax in dim=-1 which is N -> num_patches
        # shape -> [B, num_heads, N, N]
        atten_probs = atten_scores.softmax(dim=-1)
        
        if dropout is not None:
            atten_probs = dropout(atten_probs)
            
        # apply matmul on atten_porbs with V
        # atten_probs shape -> [B, num_heads, N, N]
        # V shape -> [B, num_heads, N, d_k]
        # output shape -> [B, num_heads, N, d_k]
        atten_output = torch.matmul(atten_probs, V)
        
        return atten_output
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x shape -> [B, num_heads, N, d_k]
        # we want to be -> [B, N, D]
        batch_size, num_heads, num_patches, d_k = x.shape
        x = x.transpose(2, 1).contiguous().view(batch_size, num_patches, num_heads*d_k)
        
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # create a weighted matrix for q, k, v using x
        # split the q for heads
        # shape before -> [B, N, D]
        # shape after split -> [B, num_heads, N, d_k]
        Q = self.split_heads(self.w_q(x))
        K = self.split_heads(self.w_k(x))
        V = self.split_heads(self.w_v(x))
        
        # applying self attention across q, k, v
        # shape -> [batch_size, num_heads, num_tokens, d_k]
        atten_output = self.scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            dropout=self.attention_dropout
        )
        
        comined_heads = self.combine_heads(atten_output)
        
        outputs = self.proj_dropout(self.w_o(comined_heads))
        
        # shape -> [B, N, D]
        return outputs
    
    
class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int=256,
        dff_scale: int=4,
        dropout_rate: float=0.1
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=dff_scale*embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=dff_scale*embed_dim, out_features=embed_dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape -> [B, N, 256]
        # after first fc -> [B, N, 1024]
        # after dropout -> [B, N, 1024]
        # after gelu -> [B, N, 1024]
        # after second fc -> [B, N, 256]
        out = self.mlp(x)
        
        return out
    
    
class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int=256,
        num_heads: int=8,
        dff_scale: int=4,
        dropout: float=0.1
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.atten = MultiHeadSelfAttention(
            embedding_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            dff_scale=dff_scale,
            dropout_rate=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape -> [B, N, D]
        # first apply layer norm before applying attention
        # but first save one for residual
        res1 = x
        x = self.layer_norm1(x)
        x = self.atten(x)
        # add res
        x = res1 + x
        
        res2 = x
        # apply layer norm again before mlp
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = res2 + x
        
        # shape still -> [B, N, D]
        return x
        
        
class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 256,
        num_heads: int = 8,
        dff_scale: int = 4,
        num_encoder_blocks: int = 6,
        dropout: float = 0.1,
        
    ) -> None:
        super().__init__()
        # first patch embedding
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )
        
        if image_size % patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size"
            )
        
        # second postion and cls token
        num_patches = (image_size // patch_size) ** 2
        self.class_position_embedding = AddClassAndPositionEmbedding(
            num_patches=num_patches,
            embedding_dim=embedding_dim
        )
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dff_scale=dff_scale,
                dropout=dropout
            )
            for _ in range(num_encoder_blocks)
        ])
        
        # 4. Normalize the final token representations.
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape -> [B, C, H, W]
        # apply patch embedding
        # shape -> [B, N, D]
        # print(f"image shape: {x.shape}")
        x = self.patch_embedding(x)
        # print(f"after patch embed: {x.shape}")
        # apply postion and cls token
        # shape -> [B, N+1, D]
        x = self.class_position_embedding(x)
        # print(f"after postion and cls token: {x.shape}")
        
        for block in self.encoder_blocks:
            x = block(x)
            
        # x shape -> [B, N, D]
        x = self.norm(x)
        
        # shape [B, D]
        image_features = x[:,0,:]
        # print(image_features.shape)
            
        return image_features
    
