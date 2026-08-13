"""Full encoder-decoder Transformer model."""

import torch.nn as nn


class Transformer(nn.Module):
    """Ties together Encoder + Decoder and builds the masks needed by each.

    Input:  src (batch, src_len), tgt (batch, tgt_len) token ids
    Output: (batch, tgt_len, tgt_vocab_size) logits
    """

    def __init__(self, config):
        """config: transformer.config.ModelConfig"""
        super().__init__()
        # TODO: instantiate Encoder and Decoder from config
        raise NotImplementedError

    def make_src_mask(self, src):
        # TODO: padding mask, (batch, 1, 1, src_len)
        raise NotImplementedError

    def make_tgt_mask(self, tgt):
        # TODO: padding mask & causal mask combined, (batch, 1, tgt_len, tgt_len)
        raise NotImplementedError

    def forward(self, src, tgt):
        raise NotImplementedError
