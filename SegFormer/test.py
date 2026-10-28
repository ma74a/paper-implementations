import torch

from src import (
    OverlapPatchEmbedding,
    
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


if __name__ == "__main__":
    test_overlappatchembed()