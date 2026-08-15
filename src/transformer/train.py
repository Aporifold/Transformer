"""Training entrypoint.

Usage:
    python -m transformer.train
"""

import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer.models import Transformer
from transformer.config import DataConfig, ModelConfig, TrainConfig
from transformer.data import build_dataset_splits, collate_fn
from transformer.models.transformer import Transformer


def build_dataloaders(train_cfg: TrainConfig, data_cfg: DataConfig):
    """Builds train/val DataLoaders for whichever dataset `data_cfg.name`
    selects in the registry (see `transformer.data.registry`).

    Returns `(train_set, train_loader, val_loader)`: `train_set` is returned
    too because its vocab size / special-token ids / max length are needed
    to build a matching `ModelConfig` (see `main()` below) -- this keeps
    `train.py` fully dataset-agnostic.
    """
    train_set, val_set, _ = build_dataset_splits(data_cfg.name, **data_cfg.kwargs)

    def _collate(batch):
        return collate_fn(batch, pad_idx=train_set.pad_idx)

    train_loader = DataLoader(
        train_set, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=_collate
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=_collate
    )
    return train_set, train_loader, val_loader


def build_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int) -> LambdaLR:
    """Linear warmup followed by inverse-sqrt decay, peaking at the optimizer's base lr."""

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps / step) ** 0.5

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: Transformer,
    dataloader,
    optimizer: torch.optim.Adam,
    scheduler: LambdaLR,
    criterion: nn.CrossEntropyLoss,
    train_cfg: TrainConfig,
):
    model.train()
    total_loss, n_batches = 0.0, 0

    pbar = tqdm(dataloader, desc="train", leave=False)
    for step, (src, tgt) in enumerate(pbar, start=1):
        src = src.to(train_cfg.device)
        tgt = tgt.to(train_cfg.device)

        # Teacher forcing: decoder sees tgt shifted right, predicts tgt shifted left.
        dec_input, labels = tgt[:, :-1], tgt[:, 1:]

        logits: torch.Tensor = model(src, dec_input)
        loss: torch.Tensor = criterion(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if step % train_cfg.log_every == 0:
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, train_cfg: TrainConfig):
    """No_grad pass over validation set, returns (avg_loss, token_accuracy)."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    n_correct, n_total = 0, 0
    pad_idx = model.config.pad_idx

    for src, tgt in dataloader:
        src = src.to(train_cfg.device)
        tgt = tgt.to(train_cfg.device)

        dec_input, labels = tgt[:, :-1], tgt[:, 1:]

        logits = model(src, dec_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=-1)
        mask = labels != pad_idx
        n_correct += ((preds == labels) & mask).sum().item()
        n_total += mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = n_correct / max(n_total, 1)
    return avg_loss, accuracy


def main():
    data_cfg = DataConfig()
    train_cfg = TrainConfig(
        batch_size=64,
        n_epochs=20,
        lr=3e-4,
        warmup_steps=200,
        label_smoothing=0.1,
        clip_grad_norm=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_every=20,
        ckpt_dir="checkpoints",
    )

    train_set, train_loader, val_loader = build_dataloaders(train_cfg, data_cfg)

    # Small architecture: the toy task doesn't need the full-size paper config,
    # and this keeps CPU training fast (minutes, not hours). Vocab size / pad
    # idx / max_len come from train_set, not data_cfg -- this is what lets any
    # future dataset (char-level, subword, ...) plug in without changes here.
    model_cfg = ModelConfig(
        src_vocab_size=train_set.src_vocab_size,
        tgt_vocab_size=train_set.tgt_vocab_size,
        pad_idx=train_set.pad_idx,
        max_len=train_set.max_len,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
    )
    model = Transformer(model_cfg).to(train_cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = build_lr_scheduler(optimizer, train_cfg.warmup_steps)
    criterion = nn.CrossEntropyLoss(
        ignore_index=model_cfg.pad_idx, label_smoothing=train_cfg.label_smoothing
    )

    os.makedirs(train_cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(train_cfg.ckpt_dir, "best_model.pt")
    best_val_loss = float("inf")

    print(
        f"Training on device={train_cfg.device} | dataset={data_cfg.name!r} | "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    for epoch in range(1, train_cfg.n_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, train_cfg
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, train_cfg)
        print(
            f"Epoch {epoch:02d}/{train_cfg.n_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_token_acc={val_acc:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg,
                    "data_cfg": data_cfg,
                },
                ckpt_path,
            )
            print(
                f"  -> saved new best checkpoint to '{ckpt_path}' (val_loss={val_loss:.4f})"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
