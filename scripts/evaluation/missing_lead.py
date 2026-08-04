#!/usr/bin/env python3
"""Missing-lead robustness for strict MCKI-ECG probes."""

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
from mcki_ecg.data import create_ptbxl_loaders
from mcki_ecg.evaluation import classification_metrics, load_hndr_pairs, load_linear_head


CONDITIONS = (
    "original", "random_drop_1", "random_drop_2", "random_drop_4",
    "random_drop_6", "all_limb_missing", "v1_v3_missing",
)


def apply_missing_leads(signals, condition: str, offset: int, mask_seed: int):
    if condition == "original":
        return signals
    corrupted = signals.clone()
    if condition.startswith("random_drop_"):
        count = int(condition.rsplit("_", 1)[1])
        for row in range(len(corrupted)):
            rng = np.random.default_rng(mask_seed * 1_000_003 + (offset + row) * 97 + count)
            leads = rng.choice(corrupted.shape[1], size=count, replace=False)
            corrupted[row, leads, :] = 0.0
        return corrupted
    fixed = {"all_limb_missing": [0, 1, 2, 3, 4, 5], "v1_v3_missing": [6, 7, 8]}
    corrupted[:, fixed[condition], :] = 0.0
    return corrupted


@torch.no_grad()
def infer(model, head, loader, device, condition: str, mask_seed: int):
    model.eval()
    head.eval()
    probabilities, targets, offset = [], [], 0
    for signals, labels, _ in loader:
        signals = apply_missing_leads(signals, condition, offset, mask_seed).to(device, non_blocking=True)
        features = experiment.forward_backbone_features(model, signals)
        probabilities.append(torch.sigmoid(head(features)).cpu().numpy())
        targets.append(labels.numpy())
        offset += len(labels)
    return np.concatenate(probabilities), np.concatenate(targets)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint-pattern", required=True)
    parser.add_argument("--head-pattern", required=True)
    parser.add_argument("--threshold-pattern", required=True)
    parser.add_argument("--pairs-csv", type=Path, default=Path("resources/hndr_pairs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/missing_lead"))
    parser.add_argument("--method", default="MCKI-ECG")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 1024])
    parser.add_argument("--mask-seed", type=int, default=20260710)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    pairs = load_hndr_pairs(args.pairs_csv)
    _, _, test = create_ptbxl_loaders(args.data_dir, args.batch_size, args.workers)
    loader = DataLoader(
        test.dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        encoder_path = Path(args.encoder_checkpoint_pattern.format(seed=seed))
        payload = torch.load(encoder_path, map_location="cpu", weights_only=False)
        model = experiment.build_model(copy.deepcopy(payload["cfg"]), device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        head = load_linear_head(args.head_pattern.format(seed=seed), device)
        thresholds = np.load(args.threshold_pattern.format(seed=seed))
        for condition in CONDITIONS:
            probs, targets = infer(model, head, loader, device, condition, args.mask_seed)
            run_dir = args.out_dir / f"seed_{seed}" / condition
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / "test_probs.npy", probs.astype(np.float32))
            np.save(run_dir / "test_targets.npy", targets.astype(np.float32))
            row = {
                "method": args.method, "seed": seed, "condition": condition,
                "mask_seed": args.mask_seed, "encoder_checkpoint": str(encoder_path),
                **classification_metrics(probs, targets, thresholds, pairs),
            }
            rows.append(row)
            (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        del model, head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "per_seed.csv", index=False)
    original = frame[frame.condition == "original"].set_index("seed")
    summary_rows = []
    metrics = ["Macro_AUC", "AUPRC", "Macro_F1", "MI_F1", "HNDR_Pair", "HNDR_Inst"]
    for condition, group in frame.groupby("condition", sort=False):
        row = {"method": args.method, "condition": condition, "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        decreases = np.asarray([
            float(original.loc[int(item.seed), "AUPRC"] - item.AUPRC) for item in group.itertuples()
        ])
        row["AUPRC_decrease_mean"] = float(decreases.mean())
        row["AUPRC_decrease_sd"] = float(decreases.std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
