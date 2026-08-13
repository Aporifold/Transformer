import torch
import torch.nn as nn

from .scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Module."""

    def __init__(self, d_model: int, n_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn = ScaleDotProductAttention()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # 1. QKV projection and split into H heads.
        q = self._split(self.q_proj(q))
        k = self._split(self.k_proj(k))
        v = self._split(self.v_proj(v))

        # 2. Apply scaled-dot product attention
        out, _ = self.attn(q, k, v, mask=mask)

        # 3. Concat each attention heads.
        out = self._concat(out)
        return out

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # split tensor into H heads.
        batch_size, seq_len, hidden_dim = x.size()
        out = x.view(batch_size, seq_len, self.n_heads, self.d_head)
        return out.transpose(-2, -3)

    def _concat(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, n_heads, seq_len, d_head
        batch_size, n_heads, seq_len, d_head = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
