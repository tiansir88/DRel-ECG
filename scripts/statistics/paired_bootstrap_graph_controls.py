import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
METRICS = ("AUPRC", "HNDR-Pair", "HNDR-Inst")


def hndr(probs, targets, pairs):
    pair_acc, correct, total = [], 0, 0
    for a, b in pairs:
        ia, ib = CLASS_NAMES.index(a), CLASS_NAMES.index(b)
        mask = ((targets[:, ia] >= .5) & (targets[:, ib] < .5)) | ((targets[:, ia] < .5) & (targets[:, ib] >= .5))
        if not mask.any():
            continue
        truth = (targets[mask, ia] >= .5).astype(int)
        pred = (probs[mask, ia] > probs[mask, ib]).astype(int)
        hits = pred == truth
        pair_acc.append(float(hits.mean()))
        correct += int(hits.sum())
        total += len(hits)
    if not pair_acc:
        return np.nan, np.nan
    return float(np.mean(pair_acc)), float(correct / total)


def metrics(probs, targets, pairs):
    hp, hi = hndr(probs, targets, pairs)
    return {
        "AUPRC": float(average_precision_score(targets, probs, average="macro")),
        "HNDR-Pair": hp,
        "HNDR-Inst": hi,
    }


def load_run(root, variant, seed):
    run = root / variant / f"seed_{seed}"
    return np.load(run / "test_probs.npy"), np.load(run / "test_targets.npy")


def main():
    p = argparse.ArgumentParser(description="Matched-record, matched-seed paired bootstrap for relation-graph ablations")
    p.add_argument("--root", type=Path, required=True, help="Directory containing Variant/seed_N/test_{probs,targets}.npy")
    p.add_argument("--hndr-pairs", type=Path, required=True, help="CSV with disease_a,disease_b")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 1024])
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--rng-seed", type=int, default=1729)
    args = p.parse_args()

    pair_df = pd.read_csv(args.hndr_pairs)
    pairs = [(str(r.disease_a).strip(), str(r.disease_b).strip()) for r in pair_df.itertuples()
             if str(r.disease_a).strip() in CLASS_NAMES and str(r.disease_b).strip() in CLASS_NAMES]
    comparisons = [("Hybrid_vs_Uniform_Negatives", "Uniform_Negatives"), ("Hybrid_vs_Shuffled", "Shuffled_graph")]
    rng = np.random.default_rng(args.rng_seed)
    rows, direction_rows = [], []

    for comparison, baseline in comparisons:
        runs = []
        for seed in args.seeds:
            hp, ht = load_run(args.root, "Hybrid", seed)
            bp, bt = load_run(args.root, baseline, seed)
            if hp.shape != bp.shape or ht.shape != bt.shape or not np.array_equal(ht, bt):
                raise ValueError(f"Pairing failure for {comparison}, seed {seed}: shapes/targets differ")
            hm, bm = metrics(hp, ht, pairs), metrics(bp, bt, pairs)
            runs.append((seed, hp, bp, ht, hm, bm))
            for metric in METRICS:
                delta = hm[metric] - bm[metric]
                direction_rows.append({"comparison": comparison, "seed": seed, "metric": metric,
                                       "hybrid": hm[metric], "baseline": bm[metric], "delta": delta,
                                       "positive": bool(delta > 0)})

        n = runs[0][3].shape[0]
        if any(r[3].shape[0] != n or not np.array_equal(runs[0][3], r[3]) for r in runs[1:]):
            raise ValueError(f"Targets differ across seeds for {comparison}")
        boot = {m: np.empty(args.n_boot) for m in METRICS}
        for b in range(args.n_boot):
            idx = rng.integers(0, n, n)
            per_seed = {m: [] for m in METRICS}
            for _, hp, bp, target, _, _ in runs:
                hm, bm = metrics(hp[idx], target[idx], pairs), metrics(bp[idx], target[idx], pairs)
                for metric in METRICS:
                    per_seed[metric].append(hm[metric] - bm[metric])
            for metric in METRICS:
                boot[metric][b] = np.mean(per_seed[metric])

        for metric in METRICS:
            observed = np.mean([r[4][metric] - r[5][metric] for r in runs])
            positives = sum((r[4][metric] - r[5][metric]) > 0 for r in runs)
            arr = boot[metric]
            rows.append({"comparison": comparison, "metric": metric, "delta": observed,
                         "ci_low": np.quantile(arr, .025), "ci_high": np.quantile(arr, .975),
                         "p_delta_le_0": np.mean(arr <= 0), "positive_seeds": positives,
                         "n_seeds": len(runs), "direction_consistent_positive": positives == len(runs),
                         "n_records": n, "n_boot": args.n_boot,
                         "resampling": "paired record bootstrap; mean matched-seed delta"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "paired_bootstrap_summary.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(args.out_dir / "per_seed_directions.csv", index=False)
    (args.out_dir / "run_metadata.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
