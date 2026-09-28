import torch
import torch.nn as nn
import math

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



class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float
    ):
        super(MultiHeadAttention, self).__init__()
        # check if the embed_dim is divisable by num_heads 
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.num_heads = num_heads
        # nums of embed_dim for each head (768 // 12 -> 64)
        self.d_k = embed_dim // num_heads
        # created a weighted linear for each q, k, v and ouput with dim -> [embed_dim embed_dim]
        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim) # Query 
        self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim) # Key
        self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim) # Value
        self.w_o = nn.Linear(in_features=embed_dim, out_features=embed_dim) # Output

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

    # x shape -> [batch_size, num_tokens, embed_dim]
    def split_heads(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        # shape -> [batch_size, num_tokens, num_head, d_k]
        # d_k is embed_dim // num_heads
        x = x.view(batch_size, num_tokens, self.num_heads, self.d_k)
        # then get the head before num_tokens so every token will deal with its d_k dim
        # shape -> [batch_size, num_heads, num_tokens, d_k]
        x = x.transpose(1, 2)

        # shape -> [batch_size, num_heads, num_tokens, d_k]
        return x

    def scaled_dot_product_attention(self, Q, K, V, dropout=None):
        # Q dim [batch_size, num_heads, num_tokens, d_k] is like K dim
        # we need to transpose K to match Q dim so inner dim is equal
        # K after transpose is [batch_size, num_heads, d_k, num_tokens] -> inner dim is (d_k, d_k) 
        # K.transpose(-2, -1) -> [batch_size, num_heads, d_k, num_tokens]
        # after matmul Each token gets a score against every other token
        # output shape -> [batch_size, num_heads, num_tokens, num_tokens]
        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # applying softmax dim=-1 is num_tokens
        # output shape -> [batch_size, num_heads, num_tokens, num_tokens]
        atten_probs = atten_scores.softmax(dim=-1)

        if dropout is not None:
            atten_probs = dropout(atten_probs)

        # shape -> [batch_size, num_heads, num_tokens, d_k]
        atten_output = torch.matmul(atten_probs, V)

        return atten_output

    # x shape is -> [batch_size, num_heads, num_tokens, d_k]
    # we want to be -> [batch_size, num_tokens, embed_dim]
    def combine_heads(self, x):
        batch_size, num_heads, num_tokens, d_k = x.shape
        # transpose(2, 1) make it [batch_size, num_tokens, num_heads, d_k] -> swap num_tokens with num_heads
        # view(batch_size, num_tokens, num_heads*d_k)
        x = x.transpose(2, 1).contiguous().view(batch_size, num_tokens, num_heads*d_k)

        return x



    # q, k, v shape -> [batch_size, num_tokens, embed_dim]
    def forward(self, q, k, v):

        # create weighted matrix for q, k, v
        # split the embed dim across heads
        # # shape -> [batch_size, num_heads, num_tokens, d_k] for each one
        Q = self.split_heads(self.w_q(q))
        K = self.split_heads(self.w_k(k))
        V = self.split_heads(self.w_v(v))


        # applying self attention across q, k, v
        # shape -> [batch_size, num_heads, num_tokens, d_k]
        atten_output = self.scaled_dot_product_attention(
            Q=Q,
            K=K,
            V=V,
            dropout=self.attention_dropout
        )

        # combine_heads
        # input -> [batch_size, num_heads, num_tokens, d_k]
        # output -> [batch_size, num_tokens, embed_dim]
        combined_heads = self.combine_heads(atten_output)

        outputs = self.proj_dropout(self.w_o(combined_heads))

        return outputs


class FeedForwardLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dff_scale: int,
        dropout_rate: float=0.1
    ):
        super(FeedForwardLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=dff_scale*d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=dff_scale*d_model, out_features=d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.mlp(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dff_scale: int,
        attention_dropout_rate: float,
        ff_dropout_rate: float,
    ):
        super(EncoderBlock, self).__init__()
        # The attention and FFN are two different sublayers,
        # so they should have their own normalization parameters.
        self.layer_norm_1 = LayerNormalization(embed_dim=embed_dim)
        self.layer_norm_2 = LayerNormalization(embed_dim=embed_dim)
        self.muliti_head_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=attention_dropout_rate
        )
        self.feed_forward_layer = FeedForwardLayer(
            d_model=embed_dim,
            dff_scale=dff_scale,
            dropout_rate=ff_dropout_rate
        )

    def forward(self, x):
        # First sublayer: Multi-Head Self-Attention
        residual_1 = x
        x = self.layer_norm_1(x)
        x = self.muliti_head_attention(x, x, x)
        x = residual_1 + x

        # Second sublayer: Feed Forward Network
        residual_2 = x
        x = self.feed_forward_layer(self.layer_norm_2(x))
        x = residual_2 + x

        return x

class Encoder(nn.Module):
    def __init__(
        self,
        num_encoder_blocks: int,
        embed_dim: int,
        num_heads: int,
        dff_scale: int,
        attention_dropout_rate: float,
        ff_dropout_rate: float,
    ):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dff_scale=dff_scale,
                attention_dropout_rate=attention_dropout_rate,
                ff_dropout_rate=ff_dropout_rate
            )
            for _ in range(num_encoder_blocks)
        ])

    def forward(self, x):
        for module in self.encoder_blocks:
            x = module(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channel: int,
        image_size: int,
        patch_size: int,
        embedding_dim: int,
        input_dropout_rate: float,
        num_encoder_blocks: int,
        num_heads: int,
        dff_scale: int,
        attention_dropout_rate: float,
        ff_dropout_rate: float,
    ):
        super(VisionTransformer, self).__init__()
        self.input_layer = ViTInputLayer(
            input_channel=input_channel,
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            dropout_rate=input_dropout_rate
        )
        self.encoder = Encoder(
            num_encoder_blocks=num_encoder_blocks,
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dff_scale=dff_scale,
            attention_dropout_rate=attention_dropout_rate,
            ff_dropout_rate=ff_dropout_rate
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.classification_head = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    # shape of x -> [B, C, H, W]
    def forward(self, x):
        # shape [B, N+1, D]
        x = self.input_layer(x)

        # shape [B, N+1, D]
        x = self.encoder(x)

        # applying norm
        x = self.norm(x)
        # take the class token only
        # shape [B, D]
        cls_token = x[:, 0]

        # shape [B, num_classes]
        return self.classification_head(cls_token)

        


if __name__ == "__main__":
    x = torch.randn(8, 3, 224, 224)
    model = VisionTransformer(
        num_classes=10, input_channel=3, image_size=224, patch_size=16,
        embedding_dim=768, input_dropout_rate=0.1, num_encoder_blocks=12,
        num_heads=12, dff_scale=4, attention_dropout_rate=0.0, ff_dropout_rate=0.1,
    )

    output = model(x)
    print("small output:", output.shape)                     # [8, 10]
    print("small params:", sum(p.numel() for p in model.parameters())) 
    tokens = model.input_layer(x)
    print(f"tokens: {tokens.shape}")

