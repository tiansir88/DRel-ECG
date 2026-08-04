#!/usr/bin/env python3
"""Source-only Georgia and SPH evaluation with strict MCKI-ECG probes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import mcki_ecg.experiment as experiment
from mcki_ecg.evaluation import (
    ExternalECGDataset,
    classification_metrics,
    load_hndr_pairs,
    load_linear_head,
)


@torch.no_grad()
def infer(model, head, loader, device):
    model.eval()
    head.eval()
    probabilities, targets = [], []
    for signals, labels, _ in loader:
        features = experiment.forward_backbone_features(model, signals.to(device, non_blocking=True))
        probabilities.append(torch.sigmoid(head(features)).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-checkpoint-pattern", required=True,
                        help="Pretrained encoder checkpoint path containing {seed}.")
    parser.add_argument("--head-pattern", required=True,
                        help="Strict linear-head path containing {seed}.")
    parser.add_argument("--threshold-pattern", required=True,
                        help="Validation-selected threshold path containing {seed}.")
    parser.add_argument("--georgia-dir", type=Path, required=True)
    parser.add_argument("--sph-dir", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, default=Path("resources/hndr_pairs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/external_strict_linear_probe"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 1024])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    pairs = load_hndr_pairs(args.pairs_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {"Georgia": args.georgia_dir, "SPH": args.sph_dir}
    rows = []

    for seed in args.seeds:
        encoder_path = Path(args.encoder_checkpoint_pattern.format(seed=seed))
        payload = torch.load(encoder_path, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(payload["cfg"])
        model = experiment.build_model(cfg, device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        head = load_linear_head(args.head_pattern.format(seed=seed), device)
        thresholds = np.load(args.threshold_pattern.format(seed=seed))
        for dataset_name, data_dir in datasets.items():
            loader = DataLoader(
                ExternalECGDataset(data_dir), batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda",
            )
            probs, targets = infer(model, head, loader, device)
            run_dir = args.out_dir / dataset_name.lower() / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / "test_probs.npy", probs.astype(np.float32))
            np.save(run_dir / "test_targets.npy", targets.astype(np.float32))
            np.save(run_dir / "thresholds.npy", thresholds.astype(np.float32))
            row = {
                "dataset": dataset_name, "protocol": "Strict Linear Probing source-only",
                "method": "MCKI-ECG", "seed": seed,
                "encoder_checkpoint": str(encoder_path),
                "head_checkpoint": args.head_pattern.format(seed=seed),
                **classification_metrics(probs, targets, thresholds, pairs),
            }
            rows.append(row)
            (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        del model, head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "per_seed.csv", index=False)
    summary_rows = []
    metrics = ["Macro_AUC", "AUPRC", "Macro_F1", "MI_F1", "HNDR_Pair", "HNDR_Inst"]
    for dataset_name, group in frame.groupby("dataset", sort=False):
        row = {"dataset": dataset_name, "method": "MCKI-ECG", "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
