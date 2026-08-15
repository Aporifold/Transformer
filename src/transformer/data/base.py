"""Shared contract for sequence-to-sequence datasets.

Every concrete dataset module (`toy_dataset.py`, and future modules such as
a date-conversion, arithmetic, or Multi30k dataset) should subclass
`Seq2SeqDataset` and populate the metadata attributes below. This lets
`train.py` / `inference.py` build a matching `Transformer` model purely from
the dataset instance, without any task-specific branching -- adding a new
task never requires touching either file (see `registry.py`).
"""

from typing import Any

import torch
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    """Base class every task dataset must satisfy.

    `__getitem__` must return `(src_ids, tgt_ids)` as 1-D `LongTensor`s,
    where `tgt_ids` already contains the leading `<bos>` / trailing `<eos>`.

    Subclasses set these in `__init__` (defaults below match the common
    "pad=0/bos=1/eos=2" convention used by the toy task; override if a task
    needs different ids, e.g. per-vocab special tokens for src != tgt):
    """

    src_vocab_size: int
    tgt_vocab_size: int
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    max_len: int  # longest src/tgt length the model's positional encoding must cover

    def decode(self, ids: list[int]) -> Any:
        """Turns generated token ids back into a human-readable form for
        preview logging (e.g. a string for char-level tasks). Default is a
        no-op, which is correct for integer-token tasks like the toy dataset.
        """
        return ids


def collate_fn(batch, pad_idx: int = 0):
    """Pads a list of `(src, tgt)` LongTensor pairs to the batch's max length.

    Generic across tasks: any `Seq2SeqDataset` can reuse this as long as
    `__getitem__` returns 1-D LongTensors.
    """
    srcs, tgts = zip(*batch)

    max_src_len = max(s.size(0) for s in srcs)
    max_tgt_len = max(t.size(0) for t in tgts)

    src_batch = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long)
    tgt_batch = torch.full((len(batch), max_tgt_len), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_batch[i, : s.size(0)] = s
        tgt_batch[i, : t.size(0)] = t

    return src_batch, tgt_batch
