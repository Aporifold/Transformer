import torch
import torch.nn as nn

from ..layers import LayerNorm, MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super(EncoderLayer, self).__init__()
        # sub layer 1
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        # sub layer 2
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask=None):
        residual = x
        out = self.self_attn(x, mask=src_mask)
        out = self.dropout1(out)
        out = self.norm1(out)
        out += residual

        residual = out
        out = self.ffn(out)
        out = self.dropout2(out)
        out = self.norm2(out)
        out += residual

        return out
