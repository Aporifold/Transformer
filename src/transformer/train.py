"""Training entrypoint.

Usage (once implemented):
    python -m transformer.train
"""

from transformer.config import ModelConfig, TrainConfig


def build_dataloaders(model_cfg: ModelConfig, train_cfg: TrainConfig):
    """TODO: wrap ToySeqDataset (or a real dataset later) in train/val DataLoaders."""
    raise NotImplementedError


def train_one_epoch(model, dataloader, optimizer, criterion, train_cfg: TrainConfig):
    """TODO: standard teacher-forcing training loop over one epoch, returns avg loss."""
    raise NotImplementedError


def evaluate(model, dataloader, criterion, train_cfg: TrainConfig):
    """TODO: no_grad pass over validation set, returns avg loss (+ task-specific metric)."""
    raise NotImplementedError


def main():
    # TODO: build config, model, optimizer (Adam w/ warmup schedule), loss (CrossEntropy w/ label smoothing)
    # TODO: loop over epochs calling train_one_epoch / evaluate, checkpoint best model
    raise NotImplementedError


if __name__ == "__main__":
    main()
