"""Shared evaluation utilities for MCKI-ECG experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import Dataset


CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")


class ExternalECGDataset(Dataset):
    """External array dataset with the same five-label and signal conventions."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.signals = np.load(root / "X_test.npy", mmap_mode="r")
        self.targets = np.load(root / "y_test_mh.npy", mmap_mode="r")
        if len(self.signals) != len(self.targets):
            raise ValueError("External signals and labels have different lengths")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        signal = np.asarray(self.signals[index], dtype=np.float32)
        if signal.shape == (1000, 12):
            signal = signal.T
        if signal.shape != (12, 1000):
            raise ValueError(f"Expected signal shape (12, 1000), got {signal.shape}")
        target = torch.as_tensor(np.asarray(self.targets[index]), dtype=torch.float32)
        return torch.from_numpy(signal.copy()), target, 0


def load_linear_head(path: str | Path, device: torch.device) -> nn.Linear:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("head_state_dict", payload)
    head = nn.Linear(state["weight"].shape[1], state["weight"].shape[0]).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    return head


def load_hndr_pairs(path: str | Path) -> list[tuple[str, str]]:
    frame = pd.read_csv(path)
    return [(str(row.disease_a).strip(), str(row.disease_b).strip()) for row in frame.itertuples()]


def calculate_hndr(probs: np.ndarray, targets: np.ndarray, pairs) -> tuple[float, float]:
    pair_scores, hits_total, records_total = [], 0, 0
    for first, second in pairs:
        ia, ib = CLASS_NAMES.index(first), CLASS_NAMES.index(second)
        valid = ((targets[:, ia] >= 0.5) & (targets[:, ib] < 0.5)) | (
            (targets[:, ia] < 0.5) & (targets[:, ib] >= 0.5)
        )
        if not valid.any():
            continue
        truth = targets[valid, ia] >= 0.5
        prediction = probs[valid, ia] > probs[valid, ib]
        hits = int(np.sum(truth == prediction))
        pair_scores.append(hits / int(valid.sum()))
        hits_total += hits
        records_total += int(valid.sum())
    if not pair_scores:
        raise ValueError("No evaluable HNDR class pairs")
    return float(np.mean(pair_scores)), float(hits_total / records_total)


def classification_metrics(probs, targets, thresholds, pairs) -> dict[str, float]:
    predictions = probs >= np.asarray(thresholds).reshape(1, -1)
    hndr_pair, hndr_instance = calculate_hndr(probs, targets, pairs)
    return {
        "Macro_AUC": float(roc_auc_score(targets, probs, average="macro")),
        "AUPRC": float(average_precision_score(targets, probs, average="macro")),
        "Macro_F1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "MI_F1": float(f1_score(targets[:, 1], predictions[:, 1], zero_division=0)),
        "HNDR_Pair": hndr_pair,
        "HNDR_Inst": hndr_instance,
    }
