"""Stack of N EncoderLayers, preceded by the input embedding."""

import torch.nn as nn


class Encoder(nn.Module):
    """Input:  src (batch, src_len) token ids, src_mask
    Output: (batch, src_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_len: int, pad_idx: int, dropout: float):
        super().__init__()
        # TODO: TransformerEmbedding + nn.ModuleList of n_layers EncoderLayer
        raise NotImplementedError

    def forward(self, src, src_mask):
        raise NotImplementedError
