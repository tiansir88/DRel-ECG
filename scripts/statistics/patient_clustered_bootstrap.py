#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS = [42, 123, 1024]


def hndr_pair(probs, targets, pairs):
    values = []
    for a, b in pairs:
        ia, ib = CLASS_NAMES.index(a), CLASS_NAMES.index(b)
        mask = ((targets[:, ia] >= 0.5) & (targets[:, ib] < 0.5)) | (
            (targets[:, ia] < 0.5) & (targets[:, ib] >= 0.5)
        )
        if mask.any():
            truth = targets[mask, ia] >= 0.5
            pred = probs[mask, ia] > probs[mask, ib]
            values.append(float(np.mean(pred == truth)))
    return float(np.mean(values)) if values else np.nan


def metrics(probs, targets, pairs):
    return {
        "Macro-AUC": float(roc_auc_score(targets, probs, average="macro")),
        "AUPRC": float(average_precision_score(targets, probs, average="macro")),
        "HNDR-Pair": hndr_pair(probs, targets, pairs),
    }


def load_runs(root, comparison):
    runs = []
    target_ref = None
    for seed in SEEDS:
        paths = {}
        for side in ("a", "b"):
            spec = comparison[side]
            probs = np.load(root / spec["probs"].format(seed=seed))
            targets = np.load(root / spec["targets"].format(seed=seed))
            paths[side] = (probs, targets)
        if not np.array_equal(paths["a"][1], paths["b"][1]):
            raise ValueError(f"Targets differ between methods for seed {seed}")
        if target_ref is None:
            target_ref = paths["a"][1]
        elif not np.array_equal(target_ref, paths["a"][1]):
            raise ValueError("Targets differ across seeds")
        runs.append((seed, paths["a"][0], paths["b"][0], target_ref))
    return runs, target_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--comparison-config", type=Path, required=True,
                    help="JSON list defining named a/b probability and target patterns.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=1729)
    args = ap.parse_args()

    comparisons = json.loads(args.comparison_config.read_text(encoding="utf-8"))
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("comparison-config must contain a non-empty JSON list")
    pair_df = pd.read_csv(args.pairs)
    pairs = [
        (str(r.disease_a).strip(), str(r.disease_b).strip())
        for r in pair_df.itertuples()
        if str(r.disease_a).strip() in CLASS_NAMES and str(r.disease_b).strip() in CLASS_NAMES
    ]
    manifest = pd.read_csv(args.manifest)
    patient_ids = manifest["patient_id"].to_numpy()
    unique_patients = pd.unique(patient_ids)
    groups = {pid: np.where(patient_ids == pid)[0] for pid in unique_patients}
    rows, directions = [], []

    for comp_idx, comparison in enumerate(comparisons):
        runs, targets = load_runs(args.root, comparison)
        if len(targets) != len(patient_ids):
            raise ValueError("Manifest and prediction lengths differ")
        observed = {}
        for seed, pa, pb, target in runs:
            ma, mb = metrics(pa, target, pairs), metrics(pb, target, pairs)
            for metric in ma:
                delta = ma[metric] - mb[metric]
                observed.setdefault(metric, []).append(delta)
                directions.append({
                    "comparison": comparison["name"], "seed": seed, "metric": metric,
                    "method_a": ma[metric], "method_b": mb[metric], "delta": delta,
                    "positive": bool(delta > 0),
                })

        rng = np.random.default_rng(args.rng_seed + comp_idx)
        boot = {metric: np.empty(args.n_boot, dtype=np.float64) for metric in observed}
        for b in range(args.n_boot):
            sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
            idx = np.concatenate([groups[pid] for pid in sampled])
            per_metric = {metric: [] for metric in observed}
            for _, pa, pb, target in runs:
                ma, mb = metrics(pa[idx], target[idx], pairs), metrics(pb[idx], target[idx], pairs)
                for metric in per_metric:
                    per_metric[metric].append(ma[metric] - mb[metric])
            for metric in boot:
                boot[metric][b] = np.mean(per_metric[metric])
        for metric, arr in boot.items():
            deltas = np.asarray(observed[metric])
            rows.append({
                "comparison": comparison["name"],
                "metric": metric,
                "delta": float(deltas.mean()),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "p_delta_le_0": float(np.mean(arr <= 0)),
                "positive_seeds": int(np.sum(deltas > 0)),
                "n_seeds": len(deltas),
                "n_records": len(targets),
                "n_patients": len(unique_patients),
                "n_boot": args.n_boot,
                "resampling": "patient-clustered paired bootstrap; mean matched-seed delta",
            })
        print(comparison["name"], "done", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "patient_clustered_bootstrap_summary.csv", index=False)
    pd.DataFrame(directions).to_csv(args.out_dir / "per_seed_directions.csv", index=False)
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps({
            "manifest": str(args.manifest), "pairs": str(args.pairs),
            "comparison_config": str(args.comparison_config),
            "n_boot": args.n_boot, "rng_seed": args.rng_seed,
            "n_patients": len(unique_patients), "n_records": len(patient_ids),
        }, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
