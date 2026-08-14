import torch
import torch.nn as nn

from ..blocks import TransformerDecoderLayer
from ..embedding import TransformerEmbedding


class TransformerDecoder(nn.Module):
    """Transformer Decoder."""

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
        super(TransformerDecoder, self).__init__()
        self.emb = TransformerEmbedding(vocab_size, d_model, max_len, pad_idx, dropout)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        dec: torch.Tensor,
        enc: torch.Tensor,
        tgt_mask=None,
        src_mask=None,
    ):
        out = self.emb(dec)
        for block in self.decoder_blocks:
            out = block(out, enc, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.lm_head(out)
