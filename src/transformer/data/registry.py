"""Dataset registry.

Decouples `train.py` / `inference.py` from concrete dataset implementations.
A new task is added by writing a module under `data/` that registers a
"split builder" -- a function returning `(train_set, val_set, test_set)` --
under a unique name. Nothing else in the codebase needs to change.

Example (new file `data/date_dataset.py`):

    from .registry import register_dataset

    @register_dataset("date")
    def build_date_splits(*, n_train=8000, n_val=500, n_test=200, **_):
        ...
        return train_set, val_set, test_set

Then select it via config: `DataConfig(name="date", kwargs={"n_train": ...})`.
"""

from typing import Callable

from .base import Seq2SeqDataset

SplitBuilder = Callable[..., tuple[Seq2SeqDataset, Seq2SeqDataset, Seq2SeqDataset]]

_REGISTRY: dict[str, SplitBuilder] = {}


def register_dataset(name: str):
    """Decorator that registers a `(train, val, test)` split builder under `name`."""

    def _decorator(builder: SplitBuilder) -> SplitBuilder:
        if name in _REGISTRY:
            raise ValueError(f"Dataset {name!r} is already registered")
        _REGISTRY[name] = builder
        return builder

    return _decorator


def build_dataset_splits(
    name: str, **kwargs
) -> tuple[Seq2SeqDataset, Seq2SeqDataset, Seq2SeqDataset]:
    """Returns `(train_set, val_set, test_set)` for the registered dataset `name`."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list_datasets()}")
    return _REGISTRY[name](**kwargs)


def list_datasets() -> list[str]:
    """Returns the names of all currently registered datasets."""
    return sorted(_REGISTRY)
