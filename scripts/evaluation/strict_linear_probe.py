#!/usr/bin/env python3
"""Strict linear probing for frozen DRel-ECG encoders."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

import drel_ecg.experiment as experiment
from drel_ecg.data import create_ptbxl_loaders


DEFAULT_SEEDS = (42, 123, 1024)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, *, shuffle: bool = False, seed: int = 0, workers: int = 4):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=workers, pin_memory=False, drop_last=False,
    )


@torch.no_grad()
def extract_features(model, data_loader, device):
    """Extract features with the encoder and all normalization state frozen."""

    model.eval()
    features, targets = [], []
    for signals, labels, _ in data_loader:
        encoded = experiment.forward_backbone_features(model, signals.to(device, non_blocking=True))
        features.append(encoded.cpu())
        targets.append(labels.cpu())
    return torch.cat(features).float(), torch.cat(targets).float()


@torch.no_grad()
def predict(head, features, batch_size: int = 2048):
    head.eval()
    return np.concatenate([
        torch.sigmoid(head(features[start:start + batch_size])).cpu().numpy()
        for start in range(0, len(features), batch_size)
    ])


def train_head(train_x, train_y, val_x, val_y, test_x, seed: int, cfg: dict):
    seed_everything(seed)
    head = nn.Linear(train_x.shape[1], train_y.shape[1])
    positive = train_y.sum(0)
    pos_weight = ((len(train_y) - positive) / (positive + 1e-6)).clamp(1.0, 20.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(cfg.get("lp_lr", 1e-3)),
        weight_decay=float(cfg.get("lp_weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("lp_epochs", 25))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    batches = make_loader(
        TensorDataset(train_x, train_y), int(cfg.get("batch_size", 64)),
        shuffle=True, seed=seed + 23, workers=0,
    )
    best_state, best_thresholds, best_score, best_epoch = None, None, -np.inf, -1
    patience = 0
    for epoch in range(epochs):
        head.train()
        for features, labels in batches:
            optimizer.zero_grad()
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_probs = predict(head, val_x)
        thresholds = experiment.tune_thresholds_per_class(val_probs, val_y.numpy())
        score = float(average_precision_score(val_y.numpy(), val_probs, average="macro"))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(head.state_dict())
            best_thresholds = thresholds.copy()
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        if patience >= int(cfg.get("early_stop_patience", 8)):
            break
    head.load_state_dict(best_state)
    return head, best_thresholds, best_epoch, predict(head, val_x), predict(head, test_x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", required=True,
                        help="Pretrained checkpoint path containing {seed}.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/strict_linear_probe"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_loaders = create_ptbxl_loaders(args.data_dir, args.feature_batch_size, args.workers)
    feature_loaders = [make_loader(loader.dataset, args.feature_batch_size, workers=args.workers) for loader in base_loaders]
    rows = []

    for seed in args.seeds:
        seed_everything(seed)
        checkpoint_path = Path(args.checkpoint_pattern.format(seed=seed))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = dict(experiment.CFG)
        cfg.update(checkpoint["cfg"])
        model = experiment.build_model(cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        split = [extract_features(model, loader, device) for loader in feature_loaders]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        train_x, train_y = split[0]
        val_x, val_y = split[1]
        test_x, test_y = split[2]
        head, thresholds, best_epoch, val_probs, test_probs = train_head(
            train_x, train_y, val_x, val_y, test_x, seed, cfg
        )
        run_dir = args.out_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "val_probs.npy", val_probs.astype(np.float32))
        np.save(run_dir / "val_targets.npy", val_y.numpy().astype(np.float32))
        np.save(run_dir / "test_probs.npy", test_probs.astype(np.float32))
        np.save(run_dir / "test_targets.npy", test_y.numpy().astype(np.float32))
        np.save(run_dir / "thresholds.npy", thresholds.astype(np.float32))
        torch.save({"head_state_dict": head.state_dict(), "seed": seed}, run_dir / "strict_linear_head.pt")
        metrics = experiment.evaluate_from_probs(test_probs, test_y.numpy(), thresholds)
        row = {
            "method": "DRel-ECG", "seed": seed, "protocol": "Strict Linear Probing",
            "encoder_parameters": "frozen", "normalization_state": "frozen_eval",
            "best_epoch": best_epoch, "pretrained_checkpoint": str(checkpoint_path), **metrics,
        }
        rows.append(row)
        (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "per_seed.csv", index=False)
    metric_names = ["Macro_AUC", "AUPRC", "Macro_F1", "MI_F1", "HNDR_Pair", "HNDR_Inst"]
    summary = {"method": "DRel-ECG", "protocol": "Strict Linear Probing", "n_seeds": len(frame)}
    for metric in metric_names:
        summary[f"{metric}_mean"] = float(frame[metric].mean())
        summary[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
    pd.DataFrame([summary]).to_csv(args.out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
