from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters, following the original paper's notation."""

    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_len: int = 256
    dropout: float = 0.1
    pad_idx: int = 0


@dataclass
class TrainConfig:
    """Optimization / training loop hyperparameters."""

    batch_size: int = 64
    n_epochs: int = 20
    lr: float = 1e-4
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    clip_grad_norm: float = 1.0
    device: str = "cpu"
    log_every: int = 100
    ckpt_dir: str = "checkpoints"


@dataclass
class DataConfig:
    """Selects a dataset from the registry (`transformer.data.registry`) plus
    its constructor kwargs.

    `name` must match a name registered via `@register_dataset(...)` (e.g.
    "copy"/"reverse"/"sort" from `transformer.data.toy_dataset`). `kwargs` is
    forwarded verbatim to that dataset's split builder. Adding a new task
    (date conversion, arithmetic, Multi30k, ...) means adding a new module
    under `transformer/data/` that registers itself -- this dataclass and
    `train.py`/`inference.py` never need to change.
    """

    name: str = "reverse"
    kwargs: dict = field(
        default_factory=lambda: {
            "vocab_size": 23,  # includes pad/bos/eos special tokens
            "seq_len": 16,
            "n_train": 8000,
            "n_val": 500,
            "n_test": 200,
        }
    )
