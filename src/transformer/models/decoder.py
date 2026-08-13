"""Stack of N DecoderLayers, preceded by the target embedding, followed by an output projection."""

import torch.nn as nn


class Decoder(nn.Module):
    """Input:  tgt (batch, tgt_len) token ids, enc_out, tgt_mask, src_mask
    Output: (batch, tgt_len, tgt_vocab_size) logits
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_len: int, pad_idx: int, dropout: float):
        super().__init__()
        # TODO: TransformerEmbedding + nn.ModuleList of n_layers DecoderLayer + output Linear(d_model, vocab_size)
        raise NotImplementedError

    def forward(self, tgt, enc_out, tgt_mask, src_mask):
        raise NotImplementedError
