"""Position-wise feed-forward network, section 3.3 of the paper."""

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Two linear layers with a ReLU in between, applied independently to each position.

    Input / Output: (batch, seq_len, d_model), with a d_ff-dim hidden layer in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
