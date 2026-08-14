import torch


def make_pad_mask(seq: torch.Tensor, pad_idx: int):
    """Build a padding mask: True if `id == pad_idx` else False"""
    pad_mask = seq == pad_idx
    pad_mask = pad_mask[:, None, None, :]
    return pad_mask


def make_causal_mask(seq_len: int, device=None):
    """Build a low-triangle causal mask."""
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )
    causal_mask = causal_mask[None, None, :, :].to(device=device)
    return causal_mask
