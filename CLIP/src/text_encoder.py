import torch
import torch.nn as nn


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