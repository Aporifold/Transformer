import torch
import torch.nn as nn

from ..layers import LayerNorm, MultiHeadAttention, PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super(TransformerDecoderLayer, self).__init__()
        # sub layer 1
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        # sub layer 2
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)
        # sub layer 3
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(dropout)

    def forward(
        self,
        dec: torch.Tensor,
        enc: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        residual = dec
        out = self.self_attn(q=dec, k=dec, v=dec, mask=tgt_mask)
        out = self.dropout1(out)
        out = self.norm1(out + residual)

        residual = out
        out = self.cross_attn(q=out, k=enc, v=enc, mask=src_mask)
        out = self.norm2(out + residual)

        residual = out
        out = self.ffn(out)
        out = self.norm3(out + residual)

        return out
