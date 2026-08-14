import torch
import torch.nn as nn

from ..blocks import TransformerEncoderLayer
from ..embedding import TransformerEmbedding


class TransformerEncoder(nn.Module):
    """Transformer Encoder."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        pad_idx: int,
        dropout: float,
    ):
        super(TransformerEncoder, self).__init__()
        self.emb = TransformerEmbedding(vocab_size, d_model, max_len, pad_idx, dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src: torch.Tensor, src_mask=None):
        out = self.emb(src)
        for block in self.encoder_blocks:
            out = block(out, src_mask=src_mask)
        return out
