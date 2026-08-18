#!/usr/bin/env python3
"""Georgia sensitivity analysis excluding the sparse MI class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEEP = (0, 2, 3, 4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-pattern", required=True,
                        help="Path to Georgia probabilities containing {seed}.")
    parser.add_argument("--target-pattern", required=True,
                        help="Path to Georgia targets containing {seed}.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/georgia_four_class"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 1024])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, reference = [], None
    for seed in args.seeds:
        probs_path = Path(args.prediction_pattern.format(seed=seed))
        targets_path = Path(args.target_pattern.format(seed=seed))
        probs = np.asarray(np.load(probs_path), dtype=np.float64)
        targets = np.asarray(np.load(targets_path), dtype=np.float64)
        if probs.shape != targets.shape or probs.ndim != 2 or probs.shape[1] != 5:
            raise ValueError(f"Unexpected prediction/target shapes: {probs.shape}, {targets.shape}")
        if reference is None:
            reference = targets
        elif not np.array_equal(reference, targets):
            raise ValueError("Georgia targets differ across seeds")
        probs_4c, targets_4c = probs[:, KEEP], targets[:, KEEP]
        rows.append({
            "method": "DRel-ECG", "seed": seed, "n_records": len(targets),
            "excluded_class": "MI", "included_classes": "NORM;STTC;CD;HYP",
            "Macro_AUC_4c": float(roc_auc_score(targets_4c, probs_4c, average="macro")),
            "AUPRC_4c": float(average_precision_score(targets_4c, probs_4c, average="macro")),
            "probs_path": str(probs_path), "targets_path": str(targets_path),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "per_seed.csv", index=False)
    summary = {
        "method": "DRel-ECG", "n_seeds": len(frame), "excluded_class": "MI",
        "Macro_AUC_4c_mean": float(frame.Macro_AUC_4c.mean()),
        "Macro_AUC_4c_sd": float(frame.Macro_AUC_4c.std(ddof=1)),
        "AUPRC_4c_mean": float(frame.AUPRC_4c.mean()),
        "AUPRC_4c_sd": float(frame.AUPRC_4c.std(ddof=1)),
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
