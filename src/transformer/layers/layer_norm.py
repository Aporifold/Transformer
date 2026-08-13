import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-12):
        super(LayerNorm, self).__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        raise NotImplementedError
