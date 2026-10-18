import torch
import torch.nn as nn

from image_encoder import MultiHeadSelfAttention, MLP


class TextEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int=77,
        embedding_dim: int=256
    ) -> None:
        super().__init__()
        
        # create a token embedding for shape [vocab_size, embedding_dim]
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        
        # create postion embedding for input text
        # shape [1, L, D] -> L for context length
        self.positional_embedding = nn.Parameter(
            torch.randn(1, context_length, embedding_dim) * 0.02
        )
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids shape -> [B, L]
        # after embedding -> [B, L, D]
        x = self.token_embedding(token_ids)
        
        # get the seg_len
        seq_len = token_ids.shape[1]
        
        # adding positional_embedding
        # [1, seq_len, embed_dim] + [B, seq_len, embed_dim]
        # output shape -> [B, L, D]
        x = x + self.positional_embedding[:, :seq_len, :]
        
        return x
    
class CausalMultiHeadSelfAttention(MultiHeadSelfAttention):
    """Same as Vision attention class but we'll add attention mask"""
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # x shape -> [B, L, D]
        Q = self.split_heads(self.w_q(x))  # [B, num_heads, L, d_k]
        K = self.split_heads(self.w_k(x))
        V = self.split_heads(self.w_v(x))
        
        # scores shape -> [B, num_heads, L, L]
        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # mask: True = allowed to attend, False = blocked
        # blocked positions get -inf so softmax gives them 0 probability
        if mask is not None:
            atten_scores = atten_scores.masked_fill(~mask, float("-inf"))
            
            
        atten_probs = atten_scores.softmax(dim=-1)
        atten_probs = self.attention_dropout(atten_probs)

        # [B, num_heads, L, d_k]
        atten_output = torch.matmul(atten_probs, V)

        # [B, L, D]
        combined_heads = self.combine_heads(atten_output)
        outputs = self.proj_dropout(self.w_o(combined_heads))
        
        return outputs
    
    
    
class TextEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dff_scale: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.atten = CausalMultiHeadSelfAttention(
            embedding_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            dff_scale=dff_scale,
            dropout_rate=dropout,
        )

    def forward(
        self,
        x: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # pre-norm + residual, same as the image encoder block
        x = x + self.atten(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))

        # shape still -> [B, L, D]
        return x
    
    
class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_heads: int,
        dff_scale: int,
        num_encoder_blocks: int,
        dropout: float
    ) -> None:
        super().__init__()
        self.text_embedding = TextEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim
        )
        
        self.encoder_blocks = nn.ModuleList([
            TextEncoderBlock(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dff_scale=dff_scale,
                dropout=dropout
            )
            for _ in range(num_encoder_blocks)
        ])
        
        # 4. Normalize the final token representations.
        self.norm = nn.LayerNorm(embedding_dim)
        
        # causal mask: token i can only see tokens <= i
        # shape [1, 1, L, L] so it broadcasts over batch and heads
        causal_mask = torch.tril(
            torch.ones(context_length, context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask[None, None, :, :])
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text shape -> [B, L]
        seq_len = text.shape[1]
        # pass it to text embedding
        # shape -> [B, L, D]
        x = self.text_embedding(text)
        
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for block in self.encoder_blocks:
            x = block(x, mask)

        x = self.norm(x)
        
        # With causal attention only the EOS token has seen the whole sentence.
        # EOS has the highest token id in CLIP's tokenizer, so argmax finds it.
        # shape -> [B, D]
        # get which index have eos
        eos_positions = text.argmax(dim=-1)
        # print(eos_positions)
        
        batch_indices = torch.arange(x.shape[0], device=x.device)
        # print(batch_indices)
        text_features = x[batch_indices, eos_positions]

        return text_features
    
if __name__ == "__main__":
    model = TextEncoder(
        vocab_size = 49408,
        context_length = 77,
        embedding_dim = 256,
        num_heads = 8,
        dff_scale = 4,
        num_encoder_blocks = 6,
        dropout= 0.1,
    )
    tokens = torch.randint(0, 49000, (4, 77))
    tokens[:, 10] = 49407  # fake EOS token (highest id)
    print(model(tokens).shape)  # torch.Size([4, 256])
        