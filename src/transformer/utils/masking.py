"""Mask-building helpers shared by the model and the training loop."""


def make_pad_mask(seq, pad_idx: int):
    """seq: (batch, seq_len) -> mask: (batch, 1, 1, seq_len), True where seq != pad_idx."""
    raise NotImplementedError


def make_causal_mask(seq_len: int, device=None):
    """Returns a (1, 1, seq_len, seq_len) lower-triangular mask, True = attend-allowed."""
    raise NotImplementedError
