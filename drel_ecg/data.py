"""PTB-XL array dataset used by DRel-ECG."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SEVERITY_RANK = {"MI": 5, "HYP": 4, "CD": 3, "STTC": 2, "NORM": 1}


class PTBXLDataset(Dataset):
    """Load preprocessed 10-second, 12-lead PTB-XL arrays.

    Signals may be stored as ``(N, 1000, 12)`` or ``(N, 12, 1000)`` and are
    returned as ``float32`` tensors with shape ``(12, 1000)``.  The third
    return value is a deterministic primary-label index retained for
    compatibility with the original training pipeline; GHNM itself consumes
    the complete five-dimensional multi-hot target.
    """

    def __init__(
        self,
        signal_path: str | Path,
        raw_label_path: str | Path,
        multi_hot_path: str | Path,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.signals = np.load(signal_path, mmap_mode="r")
        self.raw_labels = np.load(raw_label_path, allow_pickle=True)
        self.multi_hot = np.load(multi_hot_path, mmap_mode="r")
        if not (len(self.signals) == len(self.raw_labels) == len(self.multi_hot)):
            raise ValueError("Signal and label arrays must contain the same number of records")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.signals)

    @staticmethod
    def primary_label_index(labels) -> int:
        labels = list(labels)
        if not labels:
            return CLASS_TO_INDEX["NORM"]
        label = max(labels, key=lambda item: SEVERITY_RANK.get(str(item), 0))
        return CLASS_TO_INDEX.get(str(label), CLASS_TO_INDEX["NORM"])

    def __getitem__(self, index: int):
        signal = np.asarray(self.signals[index], dtype=np.float32)
        if signal.shape == (1000, 12):
            signal = signal.T
        if signal.shape != (12, 1000):
            raise ValueError(f"Expected signal shape (12, 1000), got {signal.shape}")
        signal_tensor = torch.from_numpy(signal.copy())
        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)
        target = torch.as_tensor(np.asarray(self.multi_hot[index]), dtype=torch.float32)
        primary = self.primary_label_index(self.raw_labels[index])
        return signal_tensor, target, primary


def create_ptbxl_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test loaders from prepared arrays."""

    data_dir = Path(data_dir)

    def dataset(split: str) -> PTBXLDataset:
        raw_name = "y_train.npy" if split == "train" else f"y_{split}_raw.npy"
        return PTBXLDataset(
            data_dir / f"X_{split}.npy",
            data_dir / raw_name,
            data_dir / f"y_{split}_mh.npy",
        )

    train = DataLoader(
        dataset("train"), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    validation = DataLoader(
        dataset("val"), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test = DataLoader(
        dataset("test"), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train, validation, test


# Backward-compatible aliases for archived checkpoints and experiment scripts.
PTBXLDatasetV3 = PTBXLDataset
create_ptbxl_loaders = create_ptbxl_loaders
