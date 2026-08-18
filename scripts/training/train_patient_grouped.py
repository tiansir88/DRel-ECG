#!/usr/bin/env python3
"""Build three patient-grouped reference graphs from true 10-epoch warm-ups."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SUPP = ROOT / "scripts" / "supplemental"
sys.path.insert(0, str(SUPP))

from scripts.training import train_component_ablations as minimal  # noqa: E402


drel = minimal.drel
crossfit = minimal.crossfit
CLASS_NAMES = list(drel.CLASS_NAMES)
SEEDS = (42, 123, 1024)


def patient_multilabel_splits(targets, patient_ids, n_folds, seed):
    """Greedy multilabel group assignment with deterministic seeded tie breaks."""
    patient_ids = np.asarray(patient_ids).astype(str)
    targets = np.asarray(targets, dtype=np.float64)
    unique, inverse = np.unique(patient_ids, return_inverse=True)
    patient_labels = np.zeros((len(unique), targets.shape[1]), dtype=np.float64)
    patient_records = np.zeros(len(unique), dtype=np.int64)
    for row, patient_idx in enumerate(inverse):
        patient_labels[patient_idx] += targets[row]
        patient_records[patient_idx] += 1

    totals = patient_labels.sum(axis=0)
    rarity = (patient_labels / np.maximum(totals, 1.0)).sum(axis=1)
    rng = np.random.default_rng(seed + 52000)
    tie = rng.random(len(unique))
    order = np.lexsort((tie, -patient_records, -rarity))
    target_labels = totals / n_folds
    target_records = patient_records.sum() / n_folds
    target_patients = len(unique) / n_folds
    fold_labels = np.zeros((n_folds, targets.shape[1]), dtype=np.float64)
    fold_records = np.zeros(n_folds, dtype=np.int64)
    fold_patients = np.zeros(n_folds, dtype=np.int64)
    assignment = np.full(len(unique), -1, dtype=np.int64)

    for step, patient_idx in enumerate(order):
        candidates = range(n_folds) if step >= n_folds else (step,)
        best = None
        for fold in candidates:
            next_labels = fold_labels[fold] + patient_labels[patient_idx]
            label_cost = np.mean(((next_labels - target_labels) / np.maximum(target_labels, 1.0)) ** 2)
            record_cost = ((fold_records[fold] + patient_records[patient_idx] - target_records) / max(target_records, 1.0)) ** 2
            patient_cost = ((fold_patients[fold] + 1 - target_patients) / max(target_patients, 1.0)) ** 2
            score = label_cost + 0.15 * record_cost + 0.05 * patient_cost
            key = (score, fold_records[fold], fold)
            if best is None or key < best[0]:
                best = (key, fold)
        fold = best[1]
        assignment[patient_idx] = fold
        fold_labels[fold] += patient_labels[patient_idx]
        fold_records[fold] += patient_records[patient_idx]
        fold_patients[fold] += 1

    record_folds = assignment[inverse]
    splits = []
    for fold in range(n_folds):
        holdout = np.flatnonzero(record_folds == fold)
        train = np.flatnonzero(record_folds != fold)
        splits.append((train, holdout))
    return splits


def raw_confusion(probs, targets):
    n = probs.shape[1]
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mask_i = (targets[:, i] >= 0.5) & (targets[:, j] < 0.5)
            mask_j = (targets[:, j] >= 0.5) & (targets[:, i] < 0.5)
            i_to_j = float(probs[mask_i, j].mean()) if mask_i.any() else 0.0
            j_to_i = float(probs[mask_j, i].mean()) if mask_j.any() else 0.0
            out[i, j] = out[j, i] = 0.5 * (i_to_j + j_to_i)
    return out


def normalize(raw):
    out = np.asarray(raw, dtype=np.float64).copy()
    np.fill_diagonal(out, 0.0)
    maximum = float(out.max())
    if maximum <= 0:
        raise ValueError("non-positive off-diagonal confusion matrix")
    out /= maximum
    np.fill_diagonal(out, 1.0)
    return out


def ranked_pairs(matrix):
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(i + 1, len(CLASS_NAMES)):
            pairs.append((float(matrix[i, j]), CLASS_NAMES[i], CLASS_NAMES[j], i, j))
    return sorted(pairs, reverse=True)


def binary_topk(matrix, k=6):
    out = np.eye(len(CLASS_NAMES), dtype=np.float32)
    for _, _, _, i, j in ranked_pairs(matrix)[:k]:
        out[i, j] = out[j, i] = 1.0
    return out


def run_seed(seed, cfg, data_dir, manifest_path, out_root, n_folds, record_root):
    seed_dir = out_root / f"seed_{seed}"
    done = seed_dir / "DONE"
    if done.exists():
        print(f"[Skip] patient-grouped reference seed={seed}", flush=True)
        return json.loads((seed_dir / "summary.json").read_text())
    seed_dir.mkdir(parents=True, exist_ok=True)
    drel.seed_everything(seed)
    device = torch.device("cuda")
    train_loader, _, _ = crossfit.create_ptbxl_loaders(
        str(data_dir), batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"])
    )
    dataset = train_loader.dataset
    targets = np.load(data_dir / "y_train_mh.npy").astype(np.float32)
    manifest = pd.read_csv(manifest_path)
    if len(dataset) != len(manifest) or len(dataset) != len(targets):
        raise ValueError(f"train length mismatch: dataset={len(dataset)} manifest={len(manifest)} targets={len(targets)}")
    if "patient_id" not in manifest:
        raise ValueError("patient_id missing from train manifest")
    patient_ids = manifest["patient_id"].astype(str).to_numpy()

    settings = minimal.build_variant_settings("Hybrid", cfg, seed)
    run_cfg = copy.deepcopy(cfg)
    run_cfg.update(settings["cfg_updates"])
    model = drel.build_model(run_cfg, device)
    warmup_epochs = int(run_cfg["warmup_pretrain_epochs"])
    warm_path = seed_dir / "warmup_checkpoint.pt"
    if warm_path.exists():
        payload = torch.load(warm_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        print(f"[Resume] loaded {warm_path}", flush=True)
    else:
        model, rhythm_projector, local_projector = drel.pretrain_with_leadaware_multiscale_relation(
            model=model,
            train_loader=train_loader,
            cfg=run_cfg,
            device=device,
            relation_matrix=np.asarray(settings["warmup_matrix"], dtype=np.float32),
            epochs=warmup_epochs,
            rhythm_projector=None,
            local_projector=None,
            stage_desc=f"patient-grouped reference warmup seed={seed}",
        )
        torch.save(
            {
                "seed": seed,
                "epoch": warmup_epochs,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "cfg": run_cfg,
                "class_names": CLASS_NAMES,
            },
            warm_path,
        )

    splits = patient_multilabel_splits(targets, patient_ids, n_folds, seed)
    oof_probs = np.zeros_like(targets, dtype=np.float32)
    oof_targets = np.zeros_like(targets, dtype=np.float32)
    fold_rows = []
    for fold_id, (train_idx, holdout_idx) in enumerate(splits):
        train_patients = set(patient_ids[train_idx])
        holdout_patients = set(patient_ids[holdout_idx])
        overlap = train_patients & holdout_patients
        if overlap:
            raise AssertionError(f"patient leakage in fold {fold_id}: {len(overlap)}")
        probs, fold_targets = crossfit.train_probe_and_predict_fold(
            model, dataset, train_idx, holdout_idx, run_cfg, device, seed, fold_id
        )
        oof_probs[holdout_idx] = probs
        oof_targets[holdout_idx] = fold_targets
        row = {
            "fold": fold_id,
            "n_train_records": len(train_idx),
            "n_holdout_records": len(holdout_idx),
            "n_train_patients": len(train_patients),
            "n_holdout_patients": len(holdout_patients),
            "patient_overlap": 0,
        }
        for class_idx, class_name in enumerate(CLASS_NAMES):
            row[f"holdout_{class_name}_positives"] = int(fold_targets[:, class_idx].sum())
        fold_rows.append(row)

    raw = raw_confusion(oof_probs, oof_targets)
    conf = normalize(raw)
    prior = drel.load_prior_matrix(run_cfg.get("relation_matrix_values")).astype(np.float64)
    fused = drel.blend_relation_matrices(prior, conf, run_cfg["lambda_prior"], run_cfg["lambda_conf"])
    final = binary_topk(fused, minimal.binary_edge_budget(run_cfg))

    np.save(seed_dir / "patient_grouped_oof_probs.npy", oof_probs)
    np.save(seed_dir / "patient_grouped_oof_targets.npy", oof_targets)
    audit_seed_dir = record_root / f"seed_{seed}"
    record_conf_path = audit_seed_dir / "confusion_normalized_matrix.csv"
    record_final_path = audit_seed_dir / "final_binary_matrix.csv"
    if record_conf_path.exists():
        record_conf = pd.read_csv(record_conf_path, index_col=0).to_numpy(dtype=np.float64)
    else:
        record_conf_path = audit_seed_dir / "S_confusion_trainonly.npy"
        record_conf = np.load(record_conf_path).astype(np.float64)
    record_fused = drel.blend_relation_matrices(prior, record_conf, run_cfg["lambda_prior"], run_cfg["lambda_conf"])
    if record_final_path.exists():
        record_final = pd.read_csv(record_final_path, index_col=0).to_numpy(dtype=np.float32)
    else:
        record_final_path = audit_seed_dir / "S_final_binary.npy"
        record_final = np.load(record_final_path).astype(np.float32)
    grouped_edges = {(a, b) for _, a, b, _, _ in ranked_pairs(fused)[:6]}
    record_edges = {(a, b) for _, a, b, _, _ in ranked_pairs(record_fused)[:6]}
    overlap = len(grouped_edges & record_edges)
    tri = np.triu_indices(len(CLASS_NAMES), 1)
    pearson = float(pearsonr(fused[tri], record_fused[tri]).statistic)
    spearman = float(spearmanr(fused[tri], record_fused[tri]).statistic)

    np.save(seed_dir / "confusion_raw_patient_grouped.npy", raw.astype(np.float32))
    np.save(seed_dir / "confusion_normalized_patient_grouped.npy", conf.astype(np.float32))
    np.save(seed_dir / "fused_patient_grouped.npy", np.asarray(fused, dtype=np.float32))
    np.save(seed_dir / "S_final_binary.npy", final)
    for name, matrix in (
        ("confusion_raw_patient_grouped.csv", raw),
        ("confusion_normalized_patient_grouped.csv", conf),
        ("fused_patient_grouped.csv", fused),
        ("S_final_binary.csv", final),
    ):
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(seed_dir / name)
    pd.DataFrame(fold_rows).to_csv(seed_dir / "patient_grouped_folds.csv", index=False)
    pair_rows = []
    record_rank = {(a, b): rank for rank, (_, a, b, _, _) in enumerate(ranked_pairs(record_fused), 1)}
    for rank, (score, a, b, _, _) in enumerate(ranked_pairs(fused), 1):
        pair_rows.append({
            "class_u": a,
            "class_v": b,
            "patient_grouped_fused": score,
            "patient_grouped_rank": rank,
            "record_fused": next(x[0] for x in ranked_pairs(record_fused) if x[1:3] == (a, b)),
            "record_rank": record_rank[(a, b)],
            "selected": int(rank <= 6),
        })
    pd.DataFrame(pair_rows).to_csv(seed_dir / "all_pairs_stability.csv", index=False)
    summary = {
        "seed": seed,
        "warmup_checkpoint": str(warm_path),
        "splitter": "greedy multilabel-stratified patient-grouped 5-fold",
        "n_records": len(dataset),
        "n_patients": int(pd.Series(patient_ids).nunique()),
        "top6_overlap": overlap,
        "edge_jaccard": overlap / (12 - overlap),
        "fused_pearson": pearson,
        "fused_spearman": spearman,
        "patient_grouped_edges": sorted(grouped_edges),
        "record_edges": sorted(record_edges),
        "binary_graph_identical": bool(np.array_equal(final, record_final)),
        "max_confusion_abs_difference": float(np.abs(conf - record_conf).max()),
    }
    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    done.write_text("done\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--manifest", type=Path, default=ROOT / "resources" / "manifests" / "ptbxl_train_manifest.csv")
    ap.add_argument("--record-root", type=Path, default=ROOT / "outputs" / "record_grouped_reference")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "patient_grouped_reference")
    ap.add_argument("--seeds", default="42,123,1024")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    cfg = copy.deepcopy(drel.CFG)
    cfg.update({"pretrain_epochs": 40, "warmup_pretrain_epochs": 10, "num_workers": 4})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_seed(seed, cfg, args.data_dir, args.manifest, args.out_dir, args.n_folds, args.record_root)
        for seed in seeds
    ]
    pd.DataFrame(summaries).to_csv(args.out_dir / "patient_grouped_stability_summary.csv", index=False)
    (args.out_dir / "DONE").write_text("done\n")


if __name__ == "__main__":
    main()
