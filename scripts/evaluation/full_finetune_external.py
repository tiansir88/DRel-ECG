#!/usr/bin/env python3
"""Matched source-only external evaluation for formal patient-grouped MCKI Full-FT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

import mcki_ecg.experiment as experiment


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
FOUR_CLASS_COLUMNS = [0, 2, 3, 4]
METRICS = ["Macro_AUC", "AUPRC", "Macro_F1", "MI_F1", "HNDR_Pair", "HNDR_Inst"]


class ExternalDataset(Dataset):
    def __init__(self, root: str):
        root_path = Path(root)
        self.x = np.load(root_path / "X_test.npy", mmap_mode="r")
        self.y = np.load(root_path / "y_test_mh.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        signal = np.asarray(self.x[index])
        if signal.shape == (1000, 12):
            signal = signal.T
        if signal.shape != (12, 1000):
            raise ValueError(f"Unexpected ECG shape {signal.shape}")
        target = torch.tensor(self.y[index], dtype=torch.float32)
        return torch.tensor(signal, dtype=torch.float32), target, torch.tensor(0, dtype=torch.long)


def calculate_hndr(probs: np.ndarray, targets: np.ndarray, pairs_csv: str) -> tuple[float, float]:
    pairs = pd.read_csv(pairs_csv)
    pair_scores: list[float] = []
    hits = 0
    total = 0
    for row in pairs.itertuples():
        a = str(row.disease_a).strip()
        b = str(row.disease_b).strip()
        if a not in CLASS_NAMES or b not in CLASS_NAMES:
            continue
        ia, ib = CLASS_NAMES.index(a), CLASS_NAMES.index(b)
        only_a = (targets[:, ia] >= 0.5) & (targets[:, ib] < 0.5)
        only_b = (targets[:, ia] < 0.5) & (targets[:, ib] >= 0.5)
        mask = only_a | only_b
        if not mask.any():
            continue
        truth = only_a[mask].astype(int)
        pred = (probs[mask, ia] > probs[mask, ib]).astype(int)
        correct = pred == truth
        pair_scores.append(float(correct.mean()))
        hits += int(correct.sum())
        total += len(correct)
    return float(np.mean(pair_scores)), float(hits / total)


def evaluate(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    pairs_csv: str,
) -> dict[str, float]:
    pred = probs >= thresholds.reshape(1, -1)
    pair, inst = calculate_hndr(probs, targets, pairs_csv)
    return {
        "Macro_AUC": float(roc_auc_score(targets, probs, average="macro")),
        "AUPRC": float(average_precision_score(targets, probs, average="macro")),
        "Macro_F1": float(f1_score(targets, pred, average="macro", zero_division=0)),
        "MI_F1": float(f1_score(targets[:, 1], pred[:, 1], zero_division=0)),
        "HNDR_Pair": pair,
        "HNDR_Inst": inst,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--georgia-dir", required=True)
    parser.add_argument("--sph-dir", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 1024])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {"Georgia": args.georgia_dir, "SPH": args.sph_dir}
    rows: list[dict] = []

    for seed in args.seeds:
        checkpoint_path = Path(args.checkpoint_pattern.format(seed=seed))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = dict(experiment.CFG)
        config.update(checkpoint["cfg"])
        model = experiment.build_model(config, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)

        for name, data_dir in datasets.items():
            loader = DataLoader(
                ExternalDataset(data_dir),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            probs, targets = experiment.collect_probs(model, loader, device)
            run_dir = output / name.lower() / f"Full_Finetune_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(run_dir / "external_probs.npy", probs)
            np.save(run_dir / "external_targets.npy", targets)
            np.save(run_dir / "thresholds.npy", thresholds)
            row = {
                "Dataset": name,
                "Protocol": "Full_Finetune_source_only",
                "Method": "MCKI-ECG patient-grouped",
                "Seed": seed,
                "Checkpoint": str(checkpoint_path),
                **evaluate(probs, targets, thresholds, args.pairs_csv),
            }
            if name == "Georgia":
                y4 = targets[:, FOUR_CLASS_COLUMNS]
                p4 = probs[:, FOUR_CLASS_COLUMNS]
                row["Macro_AUC_4c"] = float(roc_auc_score(y4, p4, average="macro"))
                row["AUPRC_4c"] = float(average_precision_score(y4, p4, average="macro"))
            (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps(row), flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output / "mcki_fullft_external_per_seed.csv", index=False)
    summary_rows: list[dict] = []
    for dataset, group in per_seed.groupby("Dataset", sort=False):
        row = {
            "Dataset": dataset,
            "Protocol": "Full_Finetune_source_only",
            "Method": "MCKI-ECG patient-grouped",
            "Num_Seeds": len(group),
        }
        columns = METRICS + (["Macro_AUC_4c", "AUPRC_4c"] if dataset == "Georgia" else [])
        for metric in columns:
            mean = float(group[metric].mean())
            std = float(group[metric].std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = f"{mean:.4f} +/- {std:.4f}"
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "mcki_fullft_external_summary.csv", index=False)
    audit = {
        "protocol": "Full fine-tuning on PTB-XL; source-only external inference with validation-selected thresholds",
        "graph_protocol": "patient-grouped relation-graph construction",
        "class_order": CLASS_NAMES,
        "georgia_four_class_columns": ["NORM", "STTC", "CD", "HYP"],
        "checkpoint_pattern": args.checkpoint_pattern,
        "seeds": args.seeds,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "PIPELINE.DONE").write_text("done\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
