import torch
import torch.nn as nn

from src import (
    PatchEmbedding,
    AddClassAndPositionEmbedding,
    MultiHeadSelfAttention
)

# test patch embedding
def test_patch_embedding():
        # Import PatchEmbedding from your module if needed.
    patch_embedding = PatchEmbedding(
        image_size=224,
        patch_size=16,
        embedding_dim=256,
    )

    # Simulate a batch of 4 RGB images.
    images = torch.randn(4, 3, 224, 224)

    patches = patch_embedding(images)

    print("Input shape:", images.shape)
    print("Output shape:", patches.shape)
    print("Number of patches:", patch_embedding.num_patches)

    assert patches.shape == (4, 196, 256)
    assert patch_embedding.num_patches == 196
    
def test_add_cls_positional():
    # Step 1: Create the patch embedding module.
    patch_embedding = PatchEmbedding(
        image_size=224,
        patch_size=16,
        embedding_dim=256,
    )

    # Step 2: Create our CLS + position embedding module.
    token_embedding = AddClassAndPositionEmbedding(
        num_patches=patch_embedding.num_patches,
        embedding_dim=256,
    )

    # Step 3: Create a dummy batch of images.
    images = torch.randn(4, 3, 224, 224)

    # Step 4: Convert images into patch tokens.
    patches = patch_embedding(images)

    # Step 5: Add CLS and positional embeddings.
    tokens = token_embedding(patches)

    print("Patches:", patches.shape)
    print("CLS token:", token_embedding.cls_token.shape)
    print("Position embeddings:", token_embedding.positional_encoding.shape)
    print("Final tokens:", tokens.shape)

    assert patches.shape == (4, 196, 256)
    assert tokens.shape == (4, 197, 256)
    assert torch.isfinite(tokens).all()

    # The first token is the CLS token after position addition.
    cls_features = tokens[:, 0, :]
    f = tokens[:, 0]
    print("CLS features:", cls_features.shape)
    print(f"F: {f.shape}")
    
def test_attention():
    attention = MultiHeadSelfAttention(
        embedding_dim=256,
        num_heads=8,
    )

    # Four images, each represented by 197 tokens.
    x = torch.randn(4, 197, 256)

    out = attention(x)

    print("Input:", x.shape)
    print("Output:", out.shape)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    
    
if __name__ == "__main__":
    # test_patch_embedding()
    # cls_token = nn.Parameter(torch.randn(1, 1, 256))
    # expanded = cls_token.expand(4, -1, -1)
    # print(expanded.shape)
    # test_add_cls_positional()
    test_attention()