"""Fixed sinusoidal positional encoding, as in section 3.5 of the paper."""

import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Precomputes sin/cos position encodings up to max_len and adds them to embeddings.

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        # TODO: build a (max_len, d_model) buffer of sin/cos position encodings
        raise NotImplementedError

    def forward(self, x):
        # TODO: slice buffer to x's seq_len and add
        raise NotImplementedError
