"""Combines token embedding + positional encoding + dropout."""

import torch.nn as nn


class TransformerEmbedding(nn.Module):
    """Input embedding block shared by encoder and decoder.

    Input:  (batch, seq_len) long tensor of token ids
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int, pad_idx: int, dropout: float):
        super().__init__()
        # TODO: instantiate TokenEmbedding + PositionalEncoding + nn.Dropout
        raise NotImplementedError

    def forward(self, x):
        # TODO: token_emb(x) + positional_encoding(x), then dropout
        raise NotImplementedError
