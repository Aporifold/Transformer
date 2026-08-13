"""A single decoder layer: masked self-attention + cross-attention + FFN."""

import torch.nn as nn


class DecoderLayer(nn.Module):
    """Input:
        dec_in:    (batch, tgt_len, d_model)
        enc_out:   (batch, src_len, d_model)
        tgt_mask:  causal + padding mask for decoder self-attention
        src_mask:  padding mask for encoder-decoder cross-attention
    Output: (batch, tgt_len, d_model)

    Sub-layers (post-LN):
        1. x = LayerNorm(x + Dropout(SelfAttention(x, x, x, tgt_mask)))
        2. x = LayerNorm(x + Dropout(CrossAttention(x, enc_out, enc_out, src_mask)))
        3. x = LayerNorm(x + Dropout(FFN(x)))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: self-attention, cross-attention, FFN, 3x LayerNorm, dropout
        raise NotImplementedError

    def forward(self, dec_in, enc_out, tgt_mask, src_mask):
        raise NotImplementedError
