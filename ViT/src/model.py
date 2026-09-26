import torch
import torch.nn as nn

class PatchCreation(nn.Module):
    def __init__(
        self,
        input_channel: int,
        patch_size: int,
        embedding_dim: int
    ) -> None:
        super(PatchCreation, self).__init__()
        self.patch_size = patch_size

        # use conv2d to split the images into patches
        self.patching_conv = nn.Conv2d(
            in_channels=input_channel,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        # then flatte the pathcing conv 
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        """
        we have an image convert it into
        14 * 14 -> 196 patches which each patch conatains of 16 * 16 * 3 -> 768
        """
        # x.shape -> [B, C, H, W]
        # last one -> W, H = W
        img_dim = x.shape[-1]

        # check if img_dim is divideable by patch size or not
        assert img_dim % self.patch_size == 0, \
        f"Given image dimension {img_dim} is not divisible by patch size {self.patch_size}"

        # check if it's one image, add a dim for batch at first
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # input x -> [B, C, H, W]
        # output x => [B, D, H/P, W/P] , D is embed dim
        # H, W will divided by patch size and became 14, 14 if patch size is 16
        x = self.patching_conv(x)

        # input x -> [B, D, H/P, W/P]
        # output x -> [B, D, N]
        # N is H/P * W/P which 14*14
        x = self.flatten(x)

        # change the order to make the N first
        # [B, N, D]
        x = x.permute(0, 2, 1)

        # Batch × Number_of_tokens × Embedding_dimension
        return x

class ViTInputLayer(nn.Module):
    def __init__(
        self,
        input_channel: int,
        image_size: int,
        patch_size: int,
        embedding_dim: int,
        dropout_rate: float
    ) -> None:
        super(ViTInputLayer, self).__init__()
        # get the patch embbeding
        self.patch_embbedings = PatchCreation(
            input_channel=input_channel,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )

        # create a cls token which will be learnable
        # will be of shape [1, 1, 768] to add to patch embedding
        # CLS = learned representation useful for classification
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim),
            requires_grad=True
        )

        # get the number of patches
        # 224 // 16 -> 14 * 14
        num_patches = (image_size // patch_size) ** 2
        # add one for cls token
        num_positions = num_patches + 1

        # create learnable positional encoding
        # shape of [1, 197, 768]
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_positions, embedding_dim),
            requires_grad=True
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape -> [B, C, H, W]
        # B
        batch_size = x.shape[0]

        # get patch embeddings 
        # input -> [B, C, H, W]
        # output -> [B, N, D]
        patch_embeddings = self.patch_embbedings(x)

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
        tokens = torch.concat((cls_token, patch_embeddings), dim=1)

        out = tokens + self.positional_encoding
        # output shape [B, N+1, D]
        return self.dropout(out)



# Unlike batch normalization, which normalizes across a batch, 
# Layer Normalization (LayerNorm) normalizes across the feature dimension for each individual example.
class LayerNormalization(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        eps: float=1e-6
    ):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        # nn.Parameter → automatically requires_grad=True by default.
        self.alpha = nn.Parameter(torch.ones(embed_dim)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(embed_dim)) # Added

    def forward(self, x):
        # LayerNorm standardizes values along the last axis (D), 
        # keeping output shape the same.
        # x shape -> [B, N+1, D]
        # -1 for D which calculate mean and std for D (features)
        # For every token, calculate the mean/std across its features
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # output shape -> (B, N+1, D)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta



