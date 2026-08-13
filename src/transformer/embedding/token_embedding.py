"""Token embedding lookup table."""

import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """Maps token ids to d_model-dim vectors.

    Input:  (batch, seq_len) long tensor of token ids
    Output: (batch, seq_len, d_model) float tensor
    """

    def __init__(self, vocab_size: int, d_model: int, pad_idx: int):
        # TODO: call super().__init__ with padding_idx=pad_idx
        raise NotImplementedError
