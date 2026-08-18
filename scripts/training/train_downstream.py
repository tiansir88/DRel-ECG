#!/usr/bin/env python3
"""Run downstream protocols from archived patient-grouped pretrained checkpoints.

This script never performs pretraining or reconstructs the graph. Each protocol
starts from the exact saved Full/Hybrid pretrained state for the requested model
seed. It writes one result per seed/protocol so parallel seed jobs do not race.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(Path.cwd()))
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument(
        "--checkpoint-root",
        default="outputs/fixed_graph_controls/Hybrid",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/downstream",
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--protocols", nargs="+", default=["Few_Shot_1%", "Few_Shot_10%", "Full_Finetune"])
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


args = parse_args()
repo = Path(args.repo_root)
supplemental = repo / "scripts" / "supplemental"
sys.path.insert(0, str(supplemental))
from scripts.training import train_fixed_graph_controls as fixed  # noqa: E402

drel = fixed.drel
crossfit = fixed.crossfit
seed = int(args.seed)
torch.set_num_threads(max(1, int(args.threads)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

checkpoint_path = Path(args.checkpoint_root) / f"seed_{seed}" / "pretrained_checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint.get("variant") != "Hybrid" or int(checkpoint.get("seed")) != seed:
    raise ValueError(f"Unexpected checkpoint identity: {checkpoint_path}")
cfg = copy.deepcopy(checkpoint["cfg"])
cfg["num_workers"] = int(args.num_workers)
if args.batch_size is not None:
    cfg["batch_size"] = int(args.batch_size)
pretrained_state = checkpoint["model_state_dict"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_train_loader, base_val_loader, base_test_loader = crossfit.create_ptbxl_loaders(
    args.data_dir,
    batch_size=int(cfg["batch_size"]),
    num_workers=int(cfg["num_workers"]),
)
train_dataset = base_train_loader.dataset
val_dataset = base_val_loader.dataset
test_dataset = base_test_loader.dataset

seed_dir = Path(args.out_dir) / "Hybrid" / f"seed_{seed}"
seed_dir.mkdir(parents=True, exist_ok=True)
metadata = {
    "seed": seed,
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_graph_protocol": checkpoint.get("graph_protocol"),
    "protocols": args.protocols,
    "device": str(device),
    "torch_threads": torch.get_num_threads(),
    "num_workers": int(cfg["num_workers"]),
    "batch_size": int(cfg["batch_size"]),
    "downstream_rng_policy": "seed_everything(model_seed) independently before each protocol",
    "pretraining_performed": False,
}
(seed_dir / "downstream_only_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

for protocol in args.protocols:
    protocol_dir = seed_dir / protocol
    result_path = protocol_dir / "result.csv"
    done_path = protocol_dir / "DONE"
    if done_path.exists() and result_path.exists():
        print(f"[Skip] seed={seed} protocol={protocol}", flush=True)
        continue

    print(f"[Start] seed={seed} protocol={protocol} device={device}", flush=True)
    drel.seed_everything(seed)
    train_loader, val_loader, test_loader, few_shot_indices = drel.prepare_protocol_loaders_from_base(
        train_dataset, val_dataset, test_dataset, cfg, protocol, seed
    )
    model = drel.build_model(cfg, device)
    model.load_state_dict(pretrained_state, strict=True)
    model, thresholds = drel.train_protocol(model, train_loader, val_loader, cfg, device, protocol)
    val_probs, val_targets = drel.collect_probs(model, val_loader, device)
    test_probs, test_targets = drel.collect_probs(model, test_loader, device)
    val_metrics = drel.evaluate_from_probs(val_probs, val_targets, thresholds)
    test_metrics = drel.evaluate_from_probs(test_probs, test_targets, thresholds)

    protocol_dir.mkdir(parents=True, exist_ok=True)
    crossfit.save_protocol_checkpoint(str(seed_dir), protocol, seed, model, thresholds, cfg)
    crossfit.save_protocol_arrays(
        str(seed_dir), protocol, val_probs, val_targets, test_probs, test_targets, thresholds
    )
    row = {
        "variant": "Hybrid",
        "seed": seed,
        "protocol": protocol,
        "n_train_samples": len(train_loader.dataset),
        "few_shot_indices": json.dumps(few_shot_indices) if few_shot_indices is not None else "",
        "graph_source": "fixed patient-grouped full DRel-ECG reference graph",
        "pretrained_checkpoint": str(checkpoint_path),
        "pretraining_performed": False,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    pd.DataFrame([row]).to_csv(result_path, index=False)
    done_path.write_text("done\n", encoding="utf-8")
    print("[Result] " + json.dumps(row), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"[Done] seed={seed}", flush=True)
