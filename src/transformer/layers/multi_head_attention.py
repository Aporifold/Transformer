"""Multi-head attention, section 3.2.2 of the paper."""

import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Projects q/k/v into n_heads subspaces, applies attention per head, concatenates and projects back.

    Input:
        q, k, v: (batch, seq_len, d_model)
        mask:    broadcastable to (batch, 1, seq_len_q, seq_len_k) or None
    Output:
        (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: linear projections for q, k, v, and output projection
        # TODO: instantiate ScaleDotProductAttention
        raise NotImplementedError

    def forward(self, q, k, v, mask=None):
        # TODO: linear project, split into heads (batch, n_heads, seq_len, d_k)
        # TODO: run attention, concat heads back to (batch, seq_len, d_model)
        # TODO: final output projection
        raise NotImplementedError
