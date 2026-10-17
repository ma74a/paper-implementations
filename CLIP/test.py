import torch
import torch.nn as nn

from src import (
    PatchEmbedding,
    AddClassAndPositionEmbedding,
    MultiHeadSelfAttention,
    EncoderBlock,
    ViTEncoder,
    TextEmbedding
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

def test_encoderblock():
    block = EncoderBlock(
        embed_dim=256,
        num_heads=8,
        dff_scale=4,
    )

    # Four images, each with 197 tokens.
    x = torch.randn(4, 197, 256, requires_grad=True)

    # Test the forward pass.
    out = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    assert out.shape == (4, 197, 256)
    assert torch.isfinite(out).all()

    # Test that gradients flow through the block.
    loss = out.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    print("Forward pass and backward pass succeeded!")
    
def test_encoder():
    
    model = ViTEncoder()

    images = torch.randn(4, 3, 224, 224)
    image_features = model(images)

    print("Input:", images.shape)
    print("Output:", image_features.shape)

    assert image_features.shape == (4, 256)
    assert torch.isfinite(image_features).all()

    print("ViT encoder test passed!")
    
def test_text_embedding():
    model = TextEmbedding(
        vocab_size=10_000,
        context_length=77,
        embedding_dim=256,
    )

    token_ids = torch.randint(low=0, high=10_000, size=(2, 5))
    print(token_ids.shape)
    output = model(token_ids)

    print(output.shape)
    # torch.Size([2, 5, 256])

    
if __name__ == "__main__":
    # test_patch_embedding()
    # cls_token = nn.Parameter(torch.randn(1, 1, 256))
    # expanded = cls_token.expand(4, -1, -1)
    # print(expanded.shape)
    # test_add_cls_positional()
    # test_attention()
    # test_encoderblock()
    # test_encoder()
    test_text_embedding()