#!/usr/bin/env python3
"""Unified train-only cross-fitted, binary edge-budget MCKI experiments.

This script is the final unified protocol:
  * S_confusion is estimated only from train-split out-of-fold predictions.
  * Diagnostic graphs are converted to binary edge-budget matched matrices.
  * All downstream protocols use the same pretrained checkpoint per seed/variant.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.training import train_crossfit_confusion as crossfit  # noqa: E402


stage8 = crossfit.stage8

DEFAULT_VARIANTS = ["Hybrid"]
DEFAULT_PROTOCOLS = ["Linear_Probing"]
DEFAULT_SEEDS = [42, 123, 1024]

COMPONENT_CFG_UPDATES = {
    "w_o_Lead_Aware_Modulation": {"use_lead_aware_input": False},
    "w_o_Dynamic_Lead_Masking": {"use_dynamic_lead_mask": False},
    "w_o_Local_Contrastive_Loss": {"local_loss_weight": 0.0},
    "w_o_Alignment": {"align_loss_weight": 0.0},
}


def _identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float32)


def _offdiag_indices(n: int):
    return np.triu_indices(n, k=1)


def _count_edges_from_threshold(mat: np.ndarray, threshold: float) -> int:
    tri = _offdiag_indices(mat.shape[0])
    return int(np.sum(np.asarray(mat)[tri] > float(threshold)))


def _binary_topk(mat: np.ndarray, edge_budget: int) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    n = arr.shape[0]
    out = np.eye(n, dtype=np.float32)
    tri = _offdiag_indices(n)
    values = arr[tri]
    edge_budget = int(max(0, min(edge_budget, len(values))))
    if edge_budget > 0:
        order = np.argsort(values)[::-1][:edge_budget]
        rows = tri[0][order]
        cols = tri[1][order]
        out[rows, cols] = 1.0
        out[cols, rows] = 1.0
    return out


def _binary_threshold(mat: np.ndarray, threshold: float) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    out = (arr > float(threshold)).astype(np.float32)
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 1.0)
    return out


def _shuffle_offdiag_symmetric(mat: np.ndarray, seed: int) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    n = arr.shape[0]
    out = np.eye(n, dtype=np.float32)
    tri = _offdiag_indices(n)
    values = arr[tri].copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    out[tri] = values
    out[(tri[1], tri[0])] = values
    return out


def _degree_matched_binary(reference: np.ndarray, seed: int) -> np.ndarray:
    """Return a different six-edge graph with the same labeled-node degrees."""
    ref = (np.asarray(reference) > 0.5).astype(np.float32)
    np.fill_diagonal(ref, 1.0)
    n = ref.shape[0]
    pairs = list(zip(*_offdiag_indices(n)))
    budget = int(sum(ref[i, j] > 0.5 for i, j in pairs))
    target_degree = (ref.sum(axis=1) - 1).astype(int)
    ref_edges = {(i, j) for i, j in pairs if ref[i, j] > 0.5}
    exact_candidates = []
    sequence_candidates = []
    for combo in itertools.combinations(pairs, budget):
        edges = set(combo)
        if edges == ref_edges:
            continue
        degree = np.zeros(n, dtype=int)
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1
        overlap = len(edges & ref_edges)
        candidate = (overlap, tuple(sorted(edges)))
        if np.array_equal(degree, target_degree):
            exact_candidates.append(candidate)
        if np.array_equal(np.sort(degree), np.sort(target_degree)):
            sequence_candidates.append(candidate)
    # Some small labeled graphs are uniquely determined by their per-node
    # degrees (as happens for the present five-node reference graph).  In that
    # case an exact degree-preserving shuffle does not exist; retain the same
    # degree multiset and record the fallback explicitly in run metadata.
    candidates = exact_candidates or sequence_candidates
    if not candidates:
        raise ValueError(f"No degree-sequence-matched graph for degrees={target_degree.tolist()}")
    minimum_overlap = min(x[0] for x in candidates)
    best = [edges for overlap, edges in candidates if overlap == minimum_overlap]
    edges = best[int(np.random.default_rng(seed + 9901).integers(len(best)))]
    out = np.eye(n, dtype=np.float32)
    for i, j in edges:
        out[i, j] = out[j, i] = 1.0
    return out


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def binary_edge_budget(cfg: Dict) -> int:
    prior = stage8.load_prior_matrix(cfg.get("relation_matrix_values"))
    ref_threshold = float(cfg.get("source_binary_reference_threshold", 0.10))
    return int(cfg.get("binary_edge_budget") or _count_edges_from_threshold(prior, ref_threshold))


def build_variant_settings(variant: str, cfg: Dict, seed: int) -> Dict:
    prior = stage8.load_prior_matrix(cfg.get("relation_matrix_values"))
    n = len(stage8.CLASS_NAMES)
    ident = _identity(n)
    ref_threshold = float(cfg.get("source_binary_reference_threshold", 0.10))
    budget = binary_edge_budget(cfg)
    prior_binary = _binary_threshold(prior, ref_threshold)
    if _count_edges_from_threshold(prior_binary, 0.5) != budget:
        prior_binary = _binary_topk(prior, budget)

    common_updates = {
        "hard_negative_threshold": 0.5,
        "use_continuous_weights": False,
        "binary_edge_budget": int(budget),
        "source_binary_reference_threshold": float(ref_threshold),
    }

    if variant == "Uniform_Negatives":
        return {
            "cfg_updates": {**common_updates, "hard_negative_threshold": 1.1, "lambda_prior": 1.0, "lambda_conf": 0.0},
            "warmup_matrix": ident,
            "final_mode": "identity",
            "shuffle_seed": None,
        }
    if variant == "Prior_only":
        return {
            "cfg_updates": {**common_updates, "lambda_prior": 1.0, "lambda_conf": 0.0},
            "warmup_matrix": prior_binary,
            "final_mode": "prior_binary",
            "shuffle_seed": None,
        }
    if variant == "Confusion_only":
        return {
            "cfg_updates": {**common_updates, "lambda_prior": 0.0, "lambda_conf": 1.0},
            "warmup_matrix": ident,
            "final_mode": "confusion_binary",
            "shuffle_seed": None,
        }
    if variant == "Shuffled_graph":
        return {
            "cfg_updates": {**common_updates, "lambda_prior": float(cfg["lambda_prior"]), "lambda_conf": float(cfg["lambda_conf"])},
            "warmup_matrix": prior_binary,
            "final_mode": "shuffled_hybrid_binary",
            "shuffle_seed": int(seed + 7701),
        }
    if variant == "Degree_Matched_Shuffled":
        return {
            "cfg_updates": {**common_updates, "lambda_prior": float(cfg["lambda_prior"]), "lambda_conf": float(cfg["lambda_conf"])},
            "warmup_matrix": prior_binary,
            "final_mode": "degree_matched_fixed_reference",
            "shuffle_seed": int(seed + 9901),
        }
    if variant == "Hybrid":
        return {
            "cfg_updates": {**common_updates, "lambda_prior": float(cfg["lambda_prior"]), "lambda_conf": float(cfg["lambda_conf"])},
            "warmup_matrix": prior_binary,
            "final_mode": "hybrid_binary",
            "shuffle_seed": None,
        }
    if variant in COMPONENT_CFG_UPDATES:
        return {
            "cfg_updates": {
                **common_updates,
                "lambda_prior": float(cfg["lambda_prior"]),
                "lambda_conf": float(cfg["lambda_conf"]),
                **COMPONENT_CFG_UPDATES[variant],
            },
            "warmup_matrix": prior_binary,
            "final_mode": "hybrid_binary",
            "shuffle_seed": None,
        }
    raise ValueError(f"Unsupported variant: {variant}")


def resolve_final_matrix(cfg: Dict, prior: np.ndarray, conf: np.ndarray, mode: str, shuffle_seed: int | None) -> np.ndarray:
    n = len(stage8.CLASS_NAMES)
    budget = binary_edge_budget(cfg)
    if mode == "identity":
        return _identity(n)
    if mode == "prior_binary":
        return _binary_topk(prior, budget)
    if mode == "confusion_binary":
        return _binary_topk(conf, budget)
    hybrid = stage8.blend_relation_matrices(prior, conf, cfg["lambda_prior"], cfg["lambda_conf"])
    if mode == "shuffled_hybrid_binary":
        if shuffle_seed is None:
            raise ValueError("shuffle_seed is required for shuffled graph")
        return _binary_topk(_shuffle_offdiag_symmetric(hybrid, shuffle_seed), budget)
    if mode == "hybrid_binary":
        return _binary_topk(hybrid, budget)
    raise ValueError(f"Unsupported final matrix mode: {mode}")


def save_binary_relation_artifacts(seed_dir: str, prior: np.ndarray, conf: np.ndarray, warmup: np.ndarray, final: np.ndarray) -> None:
    os.makedirs(seed_dir, exist_ok=True)
    np.save(os.path.join(seed_dir, "S_prior_raw.npy"), prior.astype(np.float32))
    np.save(os.path.join(seed_dir, "S_confusion_trainonly.npy"), conf.astype(np.float32))
    np.save(os.path.join(seed_dir, "S_warmup_binary.npy"), warmup.astype(np.float32))
    np.save(os.path.join(seed_dir, "S_final_binary.npy"), final.astype(np.float32))
    # Compatibility with existing readers.
    np.save(os.path.join(seed_dir, "S_prior.npy"), prior.astype(np.float32))
    np.save(os.path.join(seed_dir, "S_confusion.npy"), conf.astype(np.float32))
    np.save(os.path.join(seed_dir, "S_hybrid.npy"), final.astype(np.float32))
    with open(os.path.join(seed_dir, "relation_matrices.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": stage8.CLASS_NAMES,
                "S_prior_raw": prior.tolist(),
                "S_confusion_trainonly": conf.tolist(),
                "S_warmup_binary": warmup.tolist(),
                "S_final_binary": final.tolist(),
            },
            f,
            indent=2,
        )


def run_single_seed_variant(
    variant: str,
    seed: int,
    base_cfg: Dict,
    data_dir: str,
    out_dir: str,
    protocols: Sequence[str],
    n_folds: int,
    fixed_graph_root: str | None,
) -> List[Dict]:
    settings = build_variant_settings(variant, base_cfg, seed)
    cfg = copy.deepcopy(base_cfg)
    cfg.update(settings["cfg_updates"])

    stage8.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_train_loader, base_val_loader, base_test_loader = crossfit.create_ptbxl_loaders(
        data_dir,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
    )
    train_dataset = base_train_loader.dataset
    val_dataset = base_val_loader.dataset
    test_dataset = base_test_loader.dataset

    model = stage8.build_model(cfg, device)
    prior = stage8.load_prior_matrix(cfg.get("relation_matrix_values"))
    warmup_matrix = np.asarray(settings["warmup_matrix"], dtype=np.float32)
    warmup_epochs = min(int(cfg["warmup_pretrain_epochs"]), int(cfg["pretrain_epochs"]))
    hybrid_epochs = max(0, int(cfg["pretrain_epochs"]) - warmup_epochs)

    print(
        f"\n[Variant {variant} | Seed {seed}] train-only crossfit binary-budget | "
        f"warmup={warmup_epochs} | final={hybrid_epochs} | folds={n_folds} | budget={binary_edge_budget(cfg)}"
    )
    stage8._print_matrix_stats("S_warmup_binary", warmup_matrix)

    model, rhythm_projector, local_projector = stage8.pretrain_with_leadaware_multiscale_relation(
        model=model,
        train_loader=base_train_loader,
        cfg=cfg,
        device=device,
        relation_matrix=warmup_matrix,
        epochs=warmup_epochs,
        rhythm_projector=None,
        local_projector=None,
        stage_desc=f"{variant} warmup binary graph",
    )

    seed_dir = os.path.join(out_dir, variant, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    graph_dir = os.path.join(seed_dir, "trainonly_crossfit_graph")
    fixed_reference = None
    if fixed_graph_root and (variant in COMPONENT_CFG_UPDATES or variant in {"Hybrid", "Degree_Matched_Shuffled"}):
        fixed_path = os.path.join(fixed_graph_root, f"seed_{seed}", "S_final_binary.npy")
        fixed_reference = np.load(fixed_path).astype(np.float32)
        fixed_conf_path = os.path.join(
            fixed_graph_root, f"seed_{seed}", "confusion_normalized_patient_grouped.npy"
        )
        conf = np.load(fixed_conf_path).astype(np.float32)
        graph_summary = {
            "graph_source": "fixed patient-grouped full-MCKI reference graph",
            "fixed_graph_path": os.path.abspath(fixed_path),
            "fixed_confusion_path": os.path.abspath(fixed_conf_path),
            "n_train_records": int(len(train_dataset)),
            "n_folds": 0,
            "probe_epochs": 0,
        }
    elif settings["final_mode"] == "identity":
        conf = _identity(len(stage8.CLASS_NAMES))
        graph_summary = {
            "graph_source": "not estimated; identity graph used throughout Uniform_Negatives",
            "n_train_records": int(len(train_dataset)),
            "n_folds": 0,
            "probe_epochs": 0,
        }
    else:
        conf, graph_summary = crossfit.crossfit_trainonly_confusion_matrix(
            model,
            train_dataset,
            cfg,
            device,
            seed,
            n_folds,
            graph_dir,
        )
    if fixed_reference is not None:
        if variant == "Degree_Matched_Shuffled":
            final_matrix = _degree_matched_binary(fixed_reference, seed)
            graph_summary["reference_degree"] = (fixed_reference.sum(axis=1) - 1).astype(int).tolist()
            graph_summary["shuffled_degree"] = (final_matrix.sum(axis=1) - 1).astype(int).tolist()
            graph_summary["degree_match_type"] = (
                "exact labeled-node degree matched"
                if graph_summary["reference_degree"] == graph_summary["shuffled_degree"]
                else "degree-sequence matched fallback; exact labeled-node shuffle is non-existent"
            )
            both = int(np.sum((fixed_reference > 0.5) & (final_matrix > 0.5)))
            graph_summary["edge_overlap"] = int((both - len(stage8.CLASS_NAMES)) // 2)
        else:
            final_matrix = fixed_reference.copy()
    else:
        final_matrix = resolve_final_matrix(cfg, prior, conf, settings["final_mode"], settings["shuffle_seed"])
    stage8._print_matrix_stats("S_confusion_trainonly", conf)
    stage8._print_matrix_stats("S_final_binary", final_matrix)
    save_binary_relation_artifacts(seed_dir, prior, conf, warmup_matrix, final_matrix)

    with open(os.path.join(seed_dir, "unified_binary_budget_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": variant,
                "seed": int(seed),
                "graph_protocol": graph_summary["graph_source"] if fixed_reference is not None else "train-only 5-fold crossfit + binary edge-budget",
                "final_mode": settings["final_mode"],
                "shuffle_seed": settings["shuffle_seed"],
                "binary_edge_budget": int(binary_edge_budget(cfg)),
                "protocols": list(protocols),
                "graph_summary": graph_summary,
                "cfg": cfg,
                "warmup_epochs": int(warmup_epochs),
                "final_epochs": int(hybrid_epochs),
            },
            f,
            indent=2,
        )

    if hybrid_epochs > 0:
        model, rhythm_projector, local_projector = stage8.pretrain_with_leadaware_multiscale_relation(
            model=model,
            train_loader=base_train_loader,
            cfg=cfg,
            device=device,
            relation_matrix=final_matrix,
            epochs=hybrid_epochs,
            rhythm_projector=rhythm_projector,
            local_projector=local_projector,
            stage_desc=f"{variant} final binary graph",
        )

    pretrained_state = copy.deepcopy(model.state_dict())
    torch.save(
        {
            "variant": variant,
            "seed": int(seed),
            "graph_protocol": "train-only crossfitted binary edge-budget",
            "model_state_dict": pretrained_state,
            "cfg": cfg,
            "class_names": stage8.CLASS_NAMES,
        },
        os.path.join(seed_dir, "pretrained_checkpoint.pt"),
    )

    rows = []
    for protocol in protocols:
        proto_train_loader, proto_val_loader, proto_test_loader, few_shot_indices = stage8.prepare_protocol_loaders_from_base(
            train_dataset,
            val_dataset,
            test_dataset,
            cfg,
            protocol,
            seed,
        )
        proto_model = stage8.build_model(cfg, device)
        proto_model.load_state_dict(pretrained_state)
        proto_model, thresholds = stage8.train_protocol(
            proto_model,
            proto_train_loader,
            proto_val_loader,
            cfg,
            device,
            protocol,
        )
        val_probs, val_targets = stage8.collect_probs(proto_model, proto_val_loader, device)
        test_probs, test_targets = stage8.collect_probs(proto_model, proto_test_loader, device)
        val_metrics = stage8.evaluate_from_probs(val_probs, val_targets, thresholds)
        test_metrics = stage8.evaluate_from_probs(test_probs, test_targets, thresholds)
        crossfit.save_protocol_checkpoint(seed_dir, protocol, seed, proto_model, thresholds, cfg)
        crossfit.save_protocol_arrays(seed_dir, protocol, val_probs, val_targets, test_probs, test_targets, thresholds)
        rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "protocol": protocol,
                "n_train_samples": len(proto_train_loader.dataset),
                "few_shot_indices": json.dumps(few_shot_indices) if few_shot_indices is not None else "",
                "graph_source": graph_summary["graph_source"] if fixed_reference is not None else "train-only cross-fitted binary edge-budget",
                "n_graph_folds": int(n_folds),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
    Path(os.path.join(seed_dir, "DONE")).write_text("done\n", encoding="utf-8")
    return rows


def summarize(rows: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df.sort_values(["variant", "seed", "protocol"]).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "unified_binary_budget_per_seed_results.csv"), index=False)
    metric_cols = [
        "test_Macro_AUC",
        "test_AUPRC",
        "test_Macro_F1",
        "test_MI_F1",
        "test_HNDR_Pair",
        "test_HNDR_Inst",
    ]
    summary_rows = []
    for (variant, protocol), sub in df.groupby(["variant", "protocol"], sort=False):
        row = {"variant": variant, "protocol": protocol, "n_seeds": int(len(sub))}
        for col in metric_cols:
            name = col.replace("test_", "")
            row[f"{name}_mean"] = float(sub[col].mean())
            row[f"{name}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
            row[name] = f"{row[f'{name}_mean']:.4f} +/- {row[f'{name}_std']:.4f}"
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "unified_binary_budget_summary.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=stage8.DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fixed-graph-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--lp-epochs", type=int, default=None)
    parser.add_argument("--bootstrap-lp-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = copy.deepcopy(stage8.CFG)
    cfg["source_binary_reference_threshold"] = 0.10
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

    variants = parse_csv_strings(args.variants)
    protocols = parse_csv_strings(args.protocols)
    seeds = parse_csv_ints(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": args.data_dir,
                "out_dir": args.out_dir,
                "variants": variants,
                "protocols": protocols,
                "seeds": seeds,
                "n_folds": int(args.n_folds),
                "fixed_graph_root": args.fixed_graph_root,
                "cfg": cfg,
                "graph_protocol": "fixed patient-grouped reference graph" if args.fixed_graph_root else "train-only crossfit + binary edge-budget",
            },
            f,
            indent=2,
        )

    all_rows: List[Dict] = []
    result_path = os.path.join(args.out_dir, "unified_binary_budget_per_seed_results.csv")
    if os.path.exists(result_path):
        all_rows = pd.read_csv(result_path).to_dict("records")

    for variant in variants:
        for seed in seeds:
            done = os.path.join(args.out_dir, variant, f"seed_{seed}", "DONE")
            if os.path.exists(done):
                print(f"[Skip] {variant} seed={seed}")
                continue
            rows = run_single_seed_variant(
                variant, seed, cfg, args.data_dir, args.out_dir, protocols, int(args.n_folds), args.fixed_graph_root
            )
            if all_rows:
                old = pd.DataFrame(all_rows)
                old = old[~((old["variant"] == variant) & (old["seed"] == seed))]
                all_rows = old.to_dict("records")
            all_rows.extend(rows)
            summarize(all_rows, args.out_dir)

    if all_rows:
        summarize(all_rows, args.out_dir)
    print(f"[Done] Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
