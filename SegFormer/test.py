import torch

from src import (
    OverlapPatchEmbedding,
    EfficientSelfAttention,
    
)


def test_overlappatchembed():
    patch_embed = OverlapPatchEmbedding(
        in_channels=3,
        embed_dim=32,
        kernel_size=7,
        stride=4,
        padding=3,
    )

    image = torch.randn(2, 3, 512, 512)

    tokens, height, width = patch_embed(image)

    print("Tokens:", tokens.shape)
    print("Height:", height)
    print("Width:", width)

    assert tokens.shape == (2, 128 * 128, 32)
    assert height == 128
    assert width == 128
    
    return patch_embed

def efficient_self_attn(patch_embed):
    # Stage 1 configuration for MiT-B0
    attention = EfficientSelfAttention(
        embed_dim=32,
        num_heads=1,
        sr_ratio=8,
    )

    image = torch.randn(2, 3, 512, 512)

    tokens, height, width = patch_embed(image)
    output = attention(tokens, height, width)

    print("Input tokens:", tokens.shape)
    print("Output tokens:", output.shape)

    assert tokens.shape == (2, 16384, 32)
    assert output.shape == (2, 16384, 32)


if __name__ == "__main__":
    patch_embed = test_overlappatchembed()
    efficient_self_attn(patch_embed)