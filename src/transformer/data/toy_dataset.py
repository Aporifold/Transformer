"""
Toy Dataset: (1) Copy; (2) Reversal; and (3) Sort.
Therectically, it can achieve ~100% token accuracy if the achitecure is right.
"""

import torch

from .base import Seq2SeqDataset
from .registry import register_dataset

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
NUM_SPECIAL_TOKENS = 3

_TASKS = ("copy", "reverse", "sort")


class ToySeqDataset(Seq2SeqDataset):
    """Generates (src, tgt) pairs of random token sequences.

    E.g. for the "reverse" task: src = random tokens, tgt = reversed src
    (with <bos>/<eos> added), so the decoder must learn to attend back
    over the full source before it can emit the first output token.

    Token ids `0/1/2` are reserved for `<pad>/<bos>/<eos>`; `src` sequences
    are sampled uniformly from the remaining `[3, vocab_size)` range with a
    random length in `[min_len, seq_len]` each time, so that both the
    encoder's padding mask and the decoder's causal mask are exercised.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_samples: int,
        task: str = "reverse",
        min_len: int | None = None,
        seed: int | None = None,
    ):
        if task not in _TASKS:
            raise ValueError(f"Unknown task {task!r}, expected one of {_TASKS}")
        if vocab_size <= NUM_SPECIAL_TOKENS:
            raise ValueError(
                "vocab_size must be greater than the number of special tokens "
                f"({NUM_SPECIAL_TOKENS}), got {vocab_size}"
            )
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.task = task
        self.min_len = min_len if min_len is not None else max(1, seq_len // 2)
        self.seed = seed

        # Seq2SeqDataset metadata consumed by train.py/inference.py to build
        # a matching ModelConfig without any task-specific code.
        self.src_vocab_size = vocab_size
        self.tgt_vocab_size = vocab_size
        self.max_len = seq_len + 2  # + <bos>/<eos>

    def __len__(self):
        return self.n_samples

    def _transform(self, src: torch.Tensor) -> torch.Tensor:
        if self.task == "copy":
            return src.clone()
        if self.task == "reverse":
            return src.flip(0)
        return torch.sort(src).values  # task == "sort"

    def __getitem__(self, idx):
        """Returns a training sample (src, tgt)

        Args:
            idx (int): Batch index

        Returns:
            tuple: A tuple of LongTensor. Each tensor is of shape (B, L).
        """
        generator = (
            None if not self.seed else torch.Generator().manual_seed(self.seed + idx)
        )
        length = int(
            torch.randint(
                self.min_len, self.seq_len + 1, (1,), generator=generator
            ).item()
        )
        src = torch.randint(
            NUM_SPECIAL_TOKENS, self.vocab_size, (length,), generator=generator
        )
        core = self._transform(src)
        tgt = torch.cat([torch.tensor([BOS_IDX]), core, torch.tensor([EOS_IDX])])
        return src, tgt


def _make_split_builder(task: str):
    """Returns a registry builder that constructs train/val/test ToySeqDataset
    splits for `task`. val/test use fixed seeds so they're stable across runs.
    """

    def _build(
        *,
        vocab_size: int = 23,
        seq_len: int = 16,
        n_train: int = 8000,
        n_val: int = 500,
        n_test: int = 200,
        **_ignored,
    ):
        train_set = ToySeqDataset(vocab_size, seq_len, n_train, task=task)
        val_set = ToySeqDataset(vocab_size, seq_len, n_val, task=task, seed=42)
        test_set = ToySeqDataset(vocab_size, seq_len, n_test, task=task, seed=2026)
        return train_set, val_set, test_set

    return _build


for _task in _TASKS:
    register_dataset(_task)(_make_split_builder(_task))
