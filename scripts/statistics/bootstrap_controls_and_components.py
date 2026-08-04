#!/usr/bin/env python3
"""Patient-clustered paired bootstrap for strict matched controls/components."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS = (42, 123, 1024)
METRICS = ("Macro-AUC", "Macro-AUPRC", "HNDR-Pair", "HNDR-Inst")
COMPARATORS = (
    "Uniform_Negatives",
    "Degree_Matched_Shuffled",
    "w_o_Dynamic_Lead_Masking",
    "w_o_Local_Contrastive_Loss",
    "w_o_Alignment",
    "w_o_Lead_Aware_Modulation",
)


def hndr(probs, targets, pairs):
    values, hits_total, n_total = [], 0, 0
    for a, b in pairs:
        ia, ib = CLASS_NAMES.index(a), CLASS_NAMES.index(b)
        valid = ((targets[:, ia] >= 0.5) & (targets[:, ib] < 0.5)) | (
            (targets[:, ia] < 0.5) & (targets[:, ib] >= 0.5)
        )
        truth = targets[valid, ia] >= 0.5
        pred = probs[valid, ia] > probs[valid, ib]
        hits = int(np.sum(truth == pred))
        values.append(hits / int(valid.sum()))
        hits_total += hits
        n_total += int(valid.sum())
    return float(np.mean(values)), float(hits_total / n_total)


def metric_values(probs, targets, pairs):
    pair, inst = hndr(probs, targets, pairs)
    return {
        "Macro-AUC": float(roc_auc_score(targets, probs, average="macro")),
        "Macro-AUPRC": float(average_precision_score(targets, probs, average="macro")),
        "HNDR-Pair": pair,
        "HNDR-Inst": inst,
    }


def load_runs(root, comparator):
    runs, reference = [], None
    for seed in SEEDS:
        pa = np.load(root / "Hybrid" / f"seed_{seed}" / "test_probs.npy")
        ya = np.load(root / "Hybrid" / f"seed_{seed}" / "test_targets.npy")
        pb = np.load(root / comparator / f"seed_{seed}" / "test_probs.npy")
        yb = np.load(root / comparator / f"seed_{seed}" / "test_targets.npy")
        if not np.array_equal(ya, yb):
            raise ValueError(f"target mismatch for {comparator}, seed={seed}")
        if reference is None:
            reference = ya
        elif not np.array_equal(reference, ya):
            raise ValueError("target mismatch across seeds")
        runs.append((seed, pa, pb, ya))
    return runs, reference


def chunk_worker(payload):
    comparator, chunk_id, runs, groups, pairs, n_replicates, rng_seed = payload
    rng = np.random.default_rng(rng_seed)
    boot = {metric: np.empty(n_replicates, dtype=np.float64) for metric in METRICS}
    for replicate in range(n_replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[group] for group in sampled])
        deltas = {metric: [] for metric in METRICS}
        for _, full, ablated, targets in runs:
            a = metric_values(full[idx], targets[idx], pairs)
            b = metric_values(ablated[idx], targets[idx], pairs)
            for metric in METRICS:
                deltas[metric].append(a[metric] - b[metric])
        for metric in METRICS:
            boot[metric][replicate] = np.mean(deltas[metric])
    return comparator, chunk_id, boot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--chunks-per-comparison", type=int, default=10)
    p.add_argument("--jobs", type=int, default=36)
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest)
    patient_ids = manifest["patient_id"].astype(str).to_numpy()
    unique, inverse = np.unique(patient_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    pairs_df = pd.read_csv(args.pairs)
    pairs = [(str(r.disease_a).strip(), str(r.disease_b).strip()) for r in pairs_df.itertuples()]

    loaded, observed, per_seed = {}, {}, []
    for comparator in COMPARATORS:
        runs, targets = load_runs(args.input_root, comparator)
        if len(targets) != len(patient_ids):
            raise ValueError(f"manifest mismatch for {comparator}")
        loaded[comparator] = runs
        observed[comparator] = {metric: [] for metric in METRICS}
        for seed, full, ablated, targets in runs:
            a = metric_values(full, targets, pairs)
            b = metric_values(ablated, targets, pairs)
            for metric in METRICS:
                delta = a[metric] - b[metric]
                observed[comparator][metric].append(delta)
                per_seed.append({
                    "comparison": f"Hybrid vs {comparator}", "seed": seed,
                    "metric": metric, "Hybrid": a[metric], "comparator": b[metric],
                    "delta": delta, "positive": int(delta > 0),
                })

    base, remainder = divmod(args.n_boot, args.chunks_per_comparison)
    payloads = []
    for comp_i, comparator in enumerate(COMPARATORS):
        for chunk_id in range(args.chunks_per_comparison):
            count = base + int(chunk_id < remainder)
            payloads.append((comparator, chunk_id, loaded[comparator], groups, pairs,
                             count, 2026080400 + comp_i * 100 + chunk_id))

    chunks = {comparator: {} for comparator in COMPARATORS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(payloads))) as executor:
        futures = {executor.submit(chunk_worker, item): (item[0], item[1]) for item in payloads}
        for future in as_completed(futures):
            comparator, chunk_id, values = future.result()
            chunks[comparator][chunk_id] = values
            print(f"[done] {comparator} chunk {chunk_id + 1}/{args.chunks_per_comparison}", flush=True)

    rows = []
    for comparator in COMPARATORS:
        for metric in METRICS:
            boot = np.concatenate([chunks[comparator][i][metric] for i in range(args.chunks_per_comparison)])
            points = np.asarray(observed[comparator][metric])
            rows.append({
                "comparison": f"Hybrid vs {comparator}", "metric": metric,
                "delta": float(points.mean()), "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "nonpositive_fraction": float(np.mean(boot <= 0)),
                "positive_seeds": int(np.sum(points > 0)), "n_seeds": len(SEEDS),
                "n_records": len(patient_ids), "n_patients": len(unique), "n_boot": len(boot),
            })
    pd.DataFrame(rows).to_csv(args.out_dir / "patient_clustered_paired_bootstrap.csv", index=False)
    pd.DataFrame(per_seed).to_csv(args.out_dir / "per_seed_directions.csv", index=False)
    (args.out_dir / "metadata.json").write_text(json.dumps({
        "resampling": "patient-clustered paired bootstrap",
        "estimand": "mean of matched-seed metric differences",
        "n_boot": args.n_boot, "n_records": len(patient_ids), "n_patients": len(unique),
        "comparators": list(COMPARATORS), "chunks_per_comparison": args.chunks_per_comparison,
    }, indent=2), encoding="utf-8")
    (args.out_dir / "DONE").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
