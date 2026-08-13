"""A single encoder layer: self-attention + FFN, each with residual + LayerNorm."""

import torch.nn as nn


class EncoderLayer(nn.Module):
    """Input / Output: (batch, src_len, d_model).

    Sub-layers (post-LN, matching the original paper):
        1. x = LayerNorm(x + Dropout(SelfAttention(x, x, x, src_mask)))
        2. x = LayerNorm(x + Dropout(FFN(x)))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: self-attention, FFN, 2x LayerNorm, dropout
        raise NotImplementedError

    def forward(self, x, src_mask):
        raise NotImplementedError
