"""Greedy (or beam-search) autoregressive decoding for a trained model."""


def greedy_decode(model, src, max_len: int, bos_idx: int, eos_idx: int, device):
    """Feeds src through the encoder once, then generates tgt tokens one at a time
    until eos_idx is produced or max_len is reached.

    Returns: LongTensor (tgt_len,) of generated token ids.
    """
    raise NotImplementedError
