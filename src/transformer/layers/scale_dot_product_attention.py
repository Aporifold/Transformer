"""Scaled dot-product attention, section 3.2.1 of the paper."""

import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    """Computes softmax(QK^T / sqrt(d_k)) V with an optional mask.

    Input:
        q, k, v: (batch, n_heads, seq_len, d_k)
        mask:    (batch, 1, 1_or_seq_len, seq_len) or None, True/1 = keep, False/0 = mask out
    Output:
        out:   (batch, n_heads, seq_len, d_k)
        attn:  (batch, n_heads, seq_len, seq_len) attention weights, useful for inspection
    """

    def forward(self, q, k, v, mask=None):
        # TODO: scores = q @ k^T / sqrt(d_k)
        # TODO: apply mask (masked_fill with -inf where mask == 0)
        # TODO: softmax over last dim, then weighted sum with v
        raise NotImplementedError
