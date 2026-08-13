"""Central configuration for model architecture, training and data."""

from dataclasses import dataclass


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
