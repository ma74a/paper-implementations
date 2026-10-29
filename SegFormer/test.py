import torch

from src import (
    OverlapPatchEmbedding,
    EfficientSelfAttention,
    MixFFN,
    MiTTransformerBlock,
    MixVisionTransformer,
    
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
    
    
def test_mixfnn():
    mix_ffn = MixFFN(
        embed_dim=32,
        mlp_ratio=4,
        dropout_rate=0.0,
    )

    tokens = torch.randn(2, 128 * 128, 32)

    output = mix_ffn(
        tokens,
        H=128,
        W=128,
    )

    print("Input tokens:", tokens.shape)
    print("Output tokens:", output.shape)

    assert output.shape == (2, 16384, 32)
    

def test_mit_block(patch_embed):
    block = MiTTransformerBlock(
        embed_dim=32,
        num_heads=1,
        sr_ratio=8,
        mlp_ratio=4,
        dropout=0.0,
        drop_path=0.0,
    )

    # Generate tokens from an input image.
    image = torch.randn(2, 3, 512, 512)
    tokens, height, width = patch_embed(image)

    # Run the Transformer block.
    block.eval()

    with torch.no_grad():
        output = block(tokens, height, width)

    print("Input:", tokens.shape)
    print("Output:", output.shape)

    assert output.shape == tokens.shape
    
def test_mix_vision_transformer():
    encoder = MixVisionTransformer(
        drop_path_rate=0.1,
    )

    encoder.eval()

    image = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        features = encoder(image)

    for stage_index, feature in enumerate(features, start=1):
        print(f"Stage {stage_index}: {feature.shape}")

    # assert [tuple(f.shape) for f in features] == [
    #     (1, 32, 32, 32),
    #     (1, 64, 16, 16),
    #     (1, 160, 8, 8),
    #     (1, 256, 4, 4),
    # ]





if __name__ == "__main__":
    print("test overlap patch embedding: ")
    patch_embed = test_overlappatchembed()
    print("\ntest efficient_self_attn: ")
    efficient_self_attn(patch_embed)
    print("\ntest mix_fnn: ")
    test_mixfnn()
    print("\nTest mix transformer block: ")
    test_mit_block(patch_embed)
    print("\n Test test_mix_vision_transformer: ")
    test_mix_vision_transformer()