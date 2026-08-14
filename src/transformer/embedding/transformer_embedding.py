import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """Transformer Embedding (Token + Position Embedding)."""

    def __init__(
        self, vocab_size: int, d_model: int, max_len: int, pad_idx: int, dropout: float
    ):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, pad_idx)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        out = self.dropout(tok_emb + pos_emb)
        return out
