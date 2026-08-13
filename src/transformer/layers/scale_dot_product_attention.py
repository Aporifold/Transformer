import torch
import torch.nn as nn

from torch.nn import functional as F


class ScaleDotProductAttention(nn.Module):
    """Scaled-Dot Product Attention."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """Apply Scaled-Dot Product Attention.

        Args:
            q (torch.Tensor): Query tensor, of shape (B, H, L, D)
            k (torch.Tensor): Key tensor, of shape (B, H, L, D)
            v (torch.Tensor): Value tensor, of shape (B, H, L, D)
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            tuple (torch.Tensor, torch.Tensor): A tuple of attention outputs and attention weights
        """
        d_k = k.size(-1)
        scores: torch.Tensor = q @ k.transpose(-1, -2) * d_k**-0.5

        if mask is not None:
            scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        out = attn_weights @ v
        return out, attn_weights
