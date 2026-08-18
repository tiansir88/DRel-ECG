#!/usr/bin/env python3
"""Train-only cross-fitted confusion graph control for DRel-ECG.

This supplemental experiment avoids using the internal validation split to
estimate S_confusion. After the prior warmup, it estimates S_confusion from
out-of-fold predictions on the training split only, then runs the hybrid
pretraining stage and evaluates the requested downstream protocols.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import drel_ecg.experiment as drel  # noqa: E402
from drel_ecg.data import create_ptbxl_loaders  # noqa: E402
from drel_ecg.relation_graph import estimate_confusion_matrix_from_probs  # noqa: E402


DEFAULT_PROTOCOLS = ["Linear_Probing"]
DEFAULT_SEEDS = [42, 123, 1024]


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def fold_indices(n_items: int, n_folds: int, seed: int) -> List[np.ndarray]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > n_items:
        raise ValueError(f"n_folds={n_folds} exceeds n_items={n_items}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    return [arr.astype(np.int64) for arr in np.array_split(perm, n_folds)]


def train_probe_and_predict_fold(
    base_model: torch.nn.Module,
    train_dataset,
    train_indices: Sequence[int],
    holdout_indices: Sequence[int],
    cfg: Dict,
    device: torch.device,
    seed: int,
    fold_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 4))

    train_subset = Subset(train_dataset, list(map(int, train_indices)))
    holdout_subset = Subset(train_dataset, list(map(int, holdout_indices)))
    train_loader = drel.build_loader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed + 7100 + fold_id,
    )
    holdout_loader = drel.build_loader(
        holdout_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    probe = copy.deepcopy(base_model).to(device)
    drel.reinit_head(probe)
    drel.set_backbone_trainable(probe, False)
    optimizer = optim.AdamW(
        probe.cls_head.parameters(),
        lr=float(cfg["bootstrap_lp_lr"]),
        weight_decay=float(cfg["bootstrap_lp_weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=drel.compute_pos_weight(train_loader, device))

    epochs = int(cfg["bootstrap_lp_epochs"])
    for _ in tqdm(range(epochs), desc=f"OOF probe fold {fold_id}", leave=False):
        probe.train()
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = probe.forward_cls(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    probs, targets = drel.collect_probs(probe, holdout_loader, device)
    return probs.astype(np.float32), targets.astype(np.float32)


def crossfit_trainonly_confusion_matrix(
    warm_model: torch.nn.Module,
    train_dataset,
    cfg: Dict,
    device: torch.device,
    seed: int,
    n_folds: int,
    save_dir: str,
) -> Tuple[np.ndarray, Dict]:
    n_items = len(train_dataset)
    folds = fold_indices(n_items, n_folds, seed + 52000)
    n_classes = len(drel.CLASS_NAMES)
    oof_probs = np.zeros((n_items, n_classes), dtype=np.float32)
    oof_targets = np.zeros((n_items, n_classes), dtype=np.float32)
    fold_rows = []

    all_indices = np.arange(n_items, dtype=np.int64)
    for fold_id, holdout_idx in enumerate(folds):
        train_idx = np.setdiff1d(all_indices, holdout_idx, assume_unique=False)
        probs, targets = train_probe_and_predict_fold(
            warm_model,
            train_dataset,
            train_idx,
            holdout_idx,
            cfg,
            device,
            seed,
            fold_id,
        )
        oof_probs[holdout_idx] = probs
        oof_targets[holdout_idx] = targets
        fold_metrics = drel.evaluate_from_probs(probs, targets)
        fold_rows.append({
            "fold": fold_id,
            "n_train": int(len(train_idx)),
            "n_holdout": int(len(holdout_idx)),
            **{f"holdout_{k}": v for k, v in fold_metrics.items()},
        })

    confusion = estimate_confusion_matrix_from_probs(oof_probs, oof_targets)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "trainonly_oof_probs.npy"), oof_probs)
    np.save(os.path.join(save_dir, "trainonly_oof_targets.npy"), oof_targets)
    pd.DataFrame(fold_rows).to_csv(os.path.join(save_dir, "crossfit_probe_folds.csv"), index=False)
    oof_metrics = drel.evaluate_from_probs(oof_probs, oof_targets)
    summary = {
        "graph_source": "train-only out-of-fold predictions",
        "n_train_records": int(n_items),
        "n_folds": int(n_folds),
        "probe_epochs": int(cfg["bootstrap_lp_epochs"]),
        "probe_checkpoint_selection": "fixed final epoch; no validation split used",
        "oof_metrics": {k: float(v) for k, v in oof_metrics.items()},
    }
    with open(os.path.join(save_dir, "trainonly_crossfit_confusion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return confusion, summary


def save_protocol_arrays(
    seed_dir: str,
    protocol: str,
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    test_probs: np.ndarray,
    test_targets: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    protocol_dir = os.path.join(seed_dir, protocol)
    os.makedirs(protocol_dir, exist_ok=True)
    np.save(os.path.join(protocol_dir, "val_probs.npy"), val_probs.astype(np.float32))
    np.save(os.path.join(protocol_dir, "val_targets.npy"), val_targets.astype(np.float32))
    np.save(os.path.join(protocol_dir, "test_probs.npy"), test_probs.astype(np.float32))
    np.save(os.path.join(protocol_dir, "test_targets.npy"), test_targets.astype(np.float32))
    np.save(os.path.join(protocol_dir, "thresholds.npy"), thresholds.astype(np.float32))


def save_protocol_checkpoint(
    seed_dir: str,
    protocol: str,
    seed: int,
    model: torch.nn.Module,
    thresholds: np.ndarray,
    cfg: Dict,
) -> None:
    protocol_dir = os.path.join(seed_dir, protocol)
    os.makedirs(protocol_dir, exist_ok=True)
    torch.save(
        {
            "seed": int(seed),
            "protocol": protocol,
            "graph_source": "train-only cross-fitted",
            "model_state_dict": model.state_dict(),
            "thresholds": thresholds.astype(np.float32),
            "cfg": cfg,
            "class_names": drel.CLASS_NAMES,
        },
        os.path.join(protocol_dir, "checkpoint.pt"),
    )


def run_single_seed(
    seed: int,
    cfg: Dict,
    data_dir: str,
    save_dir: str,
    protocols: Iterable[str],
    n_folds: int,
) -> List[Dict]:
    drel.seed_everything(seed)
    drel.ensure_dir(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_train_loader, base_val_loader, base_test_loader = create_ptbxl_loaders(
        data_dir,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
    )
    train_dataset = base_train_loader.dataset
    val_dataset = base_val_loader.dataset
    test_dataset = base_test_loader.dataset

    model = drel.build_model(cfg, device)
    prior = drel.load_prior_matrix(cfg.get("relation_matrix_values"))
    warmup_epochs = min(int(cfg["warmup_pretrain_epochs"]), int(cfg["pretrain_epochs"]))
    hybrid_epochs = max(0, int(cfg["pretrain_epochs"]) - warmup_epochs)

    print(
        f"\n[Seed {seed}] train-only cross-fitted confusion graph | "
        f"warmup={warmup_epochs} | hybrid={hybrid_epochs} | folds={n_folds}"
    )
    drel._print_matrix_stats("S_prior", prior)

    model, rhythm_projector, local_projector = drel.pretrain_with_leadaware_multiscale_relation(
        model=model,
        train_loader=base_train_loader,
        cfg=cfg,
        device=device,
        relation_matrix=prior,
        epochs=warmup_epochs,
        rhythm_projector=None,
        local_projector=None,
        stage_desc="DRel-ECG warm-up prior graph",
    )

    seed_dir = os.path.join(save_dir, f"seed_{seed}")
    drel.ensure_dir(seed_dir)
    graph_dir = os.path.join(seed_dir, "trainonly_crossfit_graph")
    conf, graph_summary = crossfit_trainonly_confusion_matrix(
        model,
        train_dataset,
        cfg,
        device,
        seed,
        n_folds,
        graph_dir,
    )
    drel._print_matrix_stats("S_conf_trainonly_crossfit", conf)
    hybrid = drel.blend_relation_matrices(prior, conf, cfg["lambda_prior"], cfg["lambda_conf"])
    drel._print_matrix_stats("S_hybrid_trainonly_crossfit", hybrid)
    drel.save_relation_artifacts(seed_dir, prior, conf, hybrid)

    with open(os.path.join(seed_dir, "trainonly_crossfit_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": int(seed),
            "data_dir": data_dir,
            "protocols": list(protocols),
            "graph_summary": graph_summary,
            "cfg": cfg,
            "warmup_epochs": int(warmup_epochs),
            "hybrid_epochs": int(hybrid_epochs),
        }, f, indent=2)

    if hybrid_epochs > 0:
        model, rhythm_projector, local_projector = drel.pretrain_with_leadaware_multiscale_relation(
            model=model,
            train_loader=base_train_loader,
            cfg=cfg,
            device=device,
            relation_matrix=hybrid,
            epochs=hybrid_epochs,
            rhythm_projector=rhythm_projector,
            local_projector=local_projector,
            stage_desc="DRel-ECG hybrid train-only cross-fitted graph",
        )

    rows = []
    pretrained_state = copy.deepcopy(model.state_dict())
    torch.save(
        {
            "seed": int(seed),
            "graph_source": "train-only cross-fitted",
            "model_state_dict": pretrained_state,
            "cfg": cfg,
            "class_names": drel.CLASS_NAMES,
        },
        os.path.join(seed_dir, "pretrained_checkpoint.pt"),
    )
    for protocol in protocols:
        proto_train_loader, proto_val_loader, proto_test_loader, few_shot_indices = (
            drel.prepare_protocol_loaders_from_base(
                train_dataset,
                val_dataset,
                test_dataset,
                cfg,
                protocol,
                seed,
            )
        )
        proto_model = drel.build_model(cfg, device)
        proto_model.load_state_dict(pretrained_state)
        proto_model, thresholds = drel.train_protocol(
            proto_model,
            proto_train_loader,
            proto_val_loader,
            cfg,
            device,
            protocol,
        )
        val_probs, val_targets = drel.collect_probs(proto_model, proto_val_loader, device)
        test_probs, test_targets = drel.collect_probs(proto_model, proto_test_loader, device)
        val_metrics = drel.evaluate_from_probs(val_probs, val_targets, thresholds)
        test_metrics = drel.evaluate_from_probs(test_probs, test_targets, thresholds)
        save_protocol_checkpoint(seed_dir, protocol, seed, proto_model, thresholds, cfg)
        save_protocol_arrays(seed_dir, protocol, val_probs, val_targets, test_probs, test_targets, thresholds)
        rows.append({
            "seed": int(seed),
            "protocol": protocol,
            "n_train_samples": len(proto_train_loader.dataset),
            "few_shot_indices": json.dumps(few_shot_indices) if few_shot_indices is not None else "",
            "graph_source": "train-only cross-fitted",
            "n_graph_folds": int(n_folds),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })
    return rows


def summarize(rows: List[Dict], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "trainonly_crossfit_per_seed_results.csv"), index=False)
    metric_cols = [
        "test_Macro_AUC",
        "test_AUPRC",
        "test_Macro_F1",
        "test_MI_F1",
        "test_HNDR_Pair",
        "test_HNDR_Inst",
    ]
    summary_rows = []
    for protocol, sub in df.groupby("protocol", sort=False):
        row = {"protocol": protocol, "n_seeds": int(len(sub))}
        for col in metric_cols:
            row[col.replace("test_", "") + "_mean"] = float(sub[col].mean())
            row[col.replace("test_", "") + "_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(save_dir, "trainonly_crossfit_summary.csv"),
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=drel.DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "crossfit_confusion"))
    parser.add_argument("--seeds", default="42,123,1024")
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--lp-epochs", type=int, default=None)
    parser.add_argument("--bootstrap-lp-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = copy.deepcopy(drel.CFG)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    if args.pretrain_epochs is not None:
        cfg["pretrain_epochs"] = int(args.pretrain_epochs)
    if args.warmup_epochs is not None:
        cfg["warmup_pretrain_epochs"] = int(args.warmup_epochs)
    if args.lp_epochs is not None:
        cfg["lp_epochs"] = int(args.lp_epochs)
    if args.bootstrap_lp_epochs is not None:
        cfg["bootstrap_lp_epochs"] = int(args.bootstrap_lp_epochs)

    seeds = parse_csv_ints(args.seeds)
    protocols = parse_csv_strings(args.protocols)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data_dir": args.data_dir,
            "out_dir": args.out_dir,
            "seeds": seeds,
            "protocols": protocols,
            "n_folds": int(args.n_folds),
            "cfg": cfg,
        }, f, indent=2)

    all_rows: List[Dict] = []
    per_seed_path = os.path.join(args.out_dir, "trainonly_crossfit_per_seed_results.csv")
    for seed in seeds:
        seed_done = os.path.join(args.out_dir, f"seed_{seed}", "DONE")
        if os.path.exists(seed_done):
            print(f"[Skip] seed {seed} already done")
            if os.path.exists(per_seed_path):
                existing = pd.read_csv(per_seed_path)
                all_rows = existing.to_dict("records")
            continue
        rows = run_single_seed(seed, cfg, args.data_dir, args.out_dir, protocols, int(args.n_folds))
        all_rows.extend(rows)
        if os.path.exists(per_seed_path):
            old = pd.read_csv(per_seed_path)
            old = old[old["seed"] != seed]
            all_rows = old.to_dict("records") + rows
        summarize(all_rows, args.out_dir)
        Path(seed_done).write_text("done\n", encoding="utf-8")

    if all_rows:
        summarize(all_rows, args.out_dir)
    print(f"[Done] Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
