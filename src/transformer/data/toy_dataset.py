"""Synthetic toy task for smoke-testing the model before moving to real data.

Suggested first task: sequence copy or sequence reversal over a small
integer/character vocabulary. No download required, trains in minutes,
and makes it obvious whether masking / autoregressive decoding is correct
(the model should reach ~100% token accuracy if the architecture is right).
"""

from torch.utils.data import Dataset


class ToySeqDataset(Dataset):
    """Generates (src, tgt) pairs of random token sequences.

    E.g. for the "reverse" task: src = random tokens, tgt = reversed src
    (with <bos>/<eos> added), so the decoder must learn to attend back
    over the full source before it can emit the first output token.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int, task: str = "reverse"):
        # TODO: pre-generate or lazily generate n_samples (src, tgt) pairs
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Returns (src: LongTensor, tgt: LongTensor)."""
        raise NotImplementedError


def collate_fn(batch, pad_idx: int):
    """Pads a list of (src, tgt) pairs to the batch's max length."""
    raise NotImplementedError
