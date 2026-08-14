import torch
import torch.nn as nn

from ..config import ModelConfig
from ..utils import make_causal_mask, make_pad_mask
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder


class Transformer(nn.Module):
    """Transformer Model (Encoder-Decoder Architecture)."""

    def __init__(self, config: ModelConfig):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            vocab_size=config.src_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_len=config.max_len,
            dropout=config.dropout,
            pad_idx=config.pad_idx,
        )

        self.decoder = TransformerDecoder(
            vocab_size=config.tgt_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_len=config.max_len,
            dropout=config.dropout,
            pad_idx=config.pad_idx,
        )
        self.config = config

    def _make_src_mask(self, src: torch.Tensor):
        return make_pad_mask(src, self.config.pad_idx)

    def _make_tgt_mask(self, tgt: torch.Tensor):
        tgt_seq_len = tgt.size(1)

        pad_mask = make_pad_mask(tgt, self.config.pad_idx)
        causal_mask = make_causal_mask(tgt_seq_len, tgt.device)
        tgt_mask = pad_mask | causal_mask
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # 1. build attention masks
        src_mask = self._make_src_mask(src)
        tgt_mask = self._make_tgt_mask(tgt)

        # 2. encoder & decoder
        enc_out = self.encoder(src, src_mask=src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)

        return dec_out
