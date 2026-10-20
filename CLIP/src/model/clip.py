import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from typing import Tuple

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder

class CLIP(nn.Module):
    def __init__(
        self,
        embedding_dim: int=512,
        # image parameters
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        image_embedding_dim: int = 256,
        num_heads: int = 8,
        dff_scale: int = 4,
        num_encoder_blocks: int = 6,
        dropout: float = 0.1,
        # text parameters
        vocab_size: int=49408,
        context_length: int=77,
        text_embedding_dim: int=256,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=image_embedding_dim,
            num_heads=num_heads,
            dff_scale=dff_scale,
            num_encoder_blocks=num_encoder_blocks,
            dropout=dropout
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=text_embedding_dim,
            num_heads=num_heads,
            dff_scale=dff_scale,
            num_encoder_blocks=num_encoder_blocks,
            dropout=dropout
        )
        
        # create image, text linear projections
        self.image_projection = nn.Linear(
            in_features=image_embedding_dim,
            out_features=embedding_dim
        )
        self.text_projection = nn.Linear(
            in_features=text_embedding_dim,
            out_features=embedding_dim
        )
        
        # learnable temperature, stored in log space
        # init = log(1 / 0.07), same as the paper
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # image shape -> [B, C, H, W] -> [B, N, D]
        
        # shape -> [B, D]
        features = self.image_encoder(image)
        
        # from [B, image_dim] to [B, embed_dim]
        features = self.image_projection(features)
        
        # then we normalize it for cosine similarity
        # normalize across embedd_dim 
        features = F.normalize(features, p=2, dim=-1)
        
        # shape -> [B, D]
        return features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        # text shape -> [B, L]
        
        # shape -> [B, text_dim]
        features = self.text_encoder(text)
        
        # from [B, text_dim] to [B, embed_dim]
        features = self.text_projection(features)
        
        # then we normalize it for cosine similarity
        # normalize across embedd_dim 
        features = F.normalize(features, p=2, dim=-1)
        
        # shape -> [B, D]
        return features
    
    
    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image shape -> [B, C, H, W]
        # text shape -> [B, L]
        
        image_embedding = self.encode_image(image=image)
        text_embedding = self.encode_text(text=text)
        
        # clamp the temperature to max 100 for training stability
        logit_scale = self.logit_scale.clamp(max=math.log(100)).exp()
        
        # cosine similarity as logits
        # shape [B, B]
        logits_per_image = logit_scale * image_embedding @ text_embedding.T
        logits_per_text = logits_per_image.T
        
        
        return logits_per_image, logits_per_text
    
    
if __name__ == "__main__":
    model = CLIP()
 
    images = torch.randn(4, 3, 224, 224)
    tokens = torch.randint(low=0, high=49000, size=(4, 77))
    tokens[:, 10] = 49407  # fake EOS token (highest id)
 
    logits_per_image, logits_per_text = model(images, tokens)
    print(logits_per_image.shape)
    print(logits_per_text.shape)