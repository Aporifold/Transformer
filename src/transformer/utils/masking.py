import torch


def make_pad_mask(seq: torch.Tensor, pad_idx: int):
    """Build a padding mask."""

    pad_mask = seq == pad_idx
    pad_mask = pad_mask[:, None, None, :]
    return pad_mask


def make_causal_mask(seq_len: int, device=None):
    """Build a causal mask."""

    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    causal_mask = causal_mask[None, None, :, :]
    return causal_mask
