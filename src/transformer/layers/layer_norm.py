import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        z = (x - mean) / (var + self.eps)

        return self.gamma * z + self.beta
