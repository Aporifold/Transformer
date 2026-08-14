import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Transformer Sinusoidal Position Embeddings."""

    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._init_position_embeddings()

    def _init_position_embeddings(self):
        _2i = torch.arange(0, self.d_model, step=2)
        pos = torch.arange(self.max_len)
        inv_freqs = 1 / 10000 ** (_2i / self.d_model)
        args = pos.unsqueeze(1) * inv_freqs.unsqueeze(0)
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        pos_emb = self.pe[:seq_len].unsqueeze(0)
        return pos_emb
