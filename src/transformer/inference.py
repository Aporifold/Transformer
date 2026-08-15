"""Greedy (or beam-search) autoregressive decoding for a trained model.

Usage (once a checkpoint has been trained via `python -m transformer.train`):
    python -m transformer.inference
"""

import os

import torch

from transformer.config import DataConfig
from transformer.data import build_dataset_splits
from transformer.models.transformer import Transformer
from transformer.utils import make_causal_mask, make_pad_mask


@torch.no_grad()
def greedy_decode(model, src, max_len: int, bos_idx: int, eos_idx: int, device):
    """Feeds src through the encoder once, then generates tgt tokens one at a time
    until eos_idx is produced or max_len is reached.

    Returns: LongTensor (tgt_len,) of generated token ids.
    """
    model.eval()
    pad_idx = model.config.pad_idx

    src = src.to(device)
    if src.dim() == 1:
        src = src.unsqueeze(0)  # (1, src_len)

    src_mask = make_pad_mask(src, pad_idx)
    enc_out = model.encoder(src, src_mask=src_mask)

    ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = make_pad_mask(ys, pad_idx) | make_causal_mask(ys.size(1), device)
        dec_out = model.decoder(ys, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)

        next_token = dec_out[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if bool((next_token == eos_idx).all()):
            break

    return ys.squeeze(0)


def _strip_special(seq: torch.Tensor, bos_idx: int, eos_idx: int) -> list[int]:
    """Drops the leading <bos> and everything from the first <eos> onward."""
    tokens = seq.tolist()
    if tokens and tokens[0] == bos_idx:
        tokens = tokens[1:]
    if eos_idx in tokens:
        tokens = tokens[: tokens.index(eos_idx)]
    return tokens


def main():
    """Loads the checkpoint trained by `transformer.train` and runs greedy decoding
    over a freshly rebuilt test split, reporting exact-match accuracy.

    Rebuilds the test split via the same dataset registry used by `train.py`
    (`data_cfg.name` + `data_cfg.kwargs` saved in the checkpoint), so this
    script works unchanged for any dataset -- toy task or a future real one.
    """
    ckpt_path = os.path.join("checkpoints", "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{ckpt_path}'. Run `python -m transformer.train` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # weights_only=False: our checkpoint also pickles ModelConfig/DataConfig dataclasses
    # (not just tensor weights); safe since it is produced locally by transformer.train.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    data_cfg: DataConfig = ckpt.get("data_cfg", DataConfig())

    model = Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Different seed from train.py's validation split (seed=42) -> held-out test set.
    _, _, test_set = build_dataset_splits(data_cfg.name, **data_cfg.kwargs)

    n_correct = 0
    n_preview = 10
    print(
        f"Running greedy decoding on {len(test_set)} test samples (dataset={data_cfg.name!r})\n"
    )

    for i in range(len(test_set)):
        src, tgt = test_set[i]
        pred = greedy_decode(
            model,
            src,
            max_len=test_set.max_len,
            bos_idx=test_set.bos_idx,
            eos_idx=test_set.eos_idx,
            device=device,
        )

        pred_tokens = _strip_special(pred, test_set.bos_idx, test_set.eos_idx)
        tgt_tokens = _strip_special(tgt, test_set.bos_idx, test_set.eos_idx)
        is_correct = pred_tokens == tgt_tokens
        n_correct += int(is_correct)

        if i < n_preview:
            mark = "✓" if is_correct else "✗"
            print(f"[{mark}] src : {test_set.decode(src.tolist())}")
            print(f"     tgt : {test_set.decode(tgt_tokens)}")
            print(f"    pred : {test_set.decode(pred_tokens)}\n")

    accuracy = n_correct / len(test_set)
    print(f"Exact-match accuracy on {len(test_set)} test samples: {accuracy:.2%}")


if __name__ == "__main__":
    main()
