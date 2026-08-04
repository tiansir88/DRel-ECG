#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


def prior_from_labels(y):
    y = np.asarray(y, dtype=np.float64)
    inter = y.T @ y
    counts = y.sum(axis=0)
    union = counts[:, None] + counts[None, :] - inter
    prior = inter / np.maximum(union, 1e-12)
    np.fill_diagonal(prior, 1.0)
    return prior


def confusion_raw_from_oof(probs, targets):
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probs.shape != targets.shape or probs.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Unexpected OOF shapes: {probs.shape}, {targets.shape}")
    out = np.eye(len(CLASS_NAMES), dtype=np.float64)
    for i in range(len(CLASS_NAMES)):
        for j in range(i + 1, len(CLASS_NAMES)):
            mask_i = (targets[:, i] >= 0.5) & (targets[:, j] < 0.5)
            mask_j = (targets[:, j] >= 0.5) & (targets[:, i] < 0.5)
            i_to_j = float(probs[mask_i, j].mean()) if mask_i.any() else 0.0
            j_to_i = float(probs[mask_j, i].mean()) if mask_j.any() else 0.0
            out[i, j] = out[j, i] = 0.5 * (i_to_j + j_to_i)
    return out


def normalize_confusion(raw):
    out = np.asarray(raw, dtype=np.float64).copy()
    np.fill_diagonal(out, 0.0)
    max_offdiag = float(out.max())
    if max_offdiag <= 0:
        raise ValueError("Confusion matrix has no positive off-diagonal entries")
    out /= max_offdiag
    np.fill_diagonal(out, 1.0)
    return out, max_offdiag


def matrix_csv(matrix, path):
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(path, index_label="class")


def top_edges(prior, raw, normalized, fused, k):
    rows = []
    for i in range(len(CLASS_NAMES)):
        for j in range(i + 1, len(CLASS_NAMES)):
            rows.append({
                "class_u": CLASS_NAMES[i], "class_v": CLASS_NAMES[j],
                "prior_score": prior[i, j], "confusion_raw_score": raw[i, j],
                "confusion_normalized_score": normalized[i, j],
                "fused_score": fused[i, j], "i": i, "j": j,
            })
    rows.sort(key=lambda x: x["fused_score"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["retained"] = rank <= k
    return rows


def heatmap(ax, matrix, cmap, title, decimals):
    shown = matrix.copy()
    np.fill_diagonal(shown, np.nan)
    # QuadMesh keeps the heatmap cells as PDF vector paths; imshow would embed
    # raster images and would therefore not produce a genuinely vector figure.
    boundaries = np.arange(len(CLASS_NAMES) + 1, dtype=float) - 0.5
    image = ax.pcolormesh(
        boundaries,
        boundaries,
        shown,
        vmin=0,
        vmax=1,
        cmap=cmap,
        shading="flat",
        rasterized=False,
    )
    ax.set_xlim(-0.5, len(CLASS_NAMES) - 0.5)
    ax.set_ylim(len(CLASS_NAMES) - 0.5, -0.5)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.xaxis.tick_top()
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            text = "-" if i == j else f"{matrix[i, j]:.{decimals}f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color="#202020")
    ax.set_title(title, pad=24, fontsize=11)
    ax.tick_params(length=0, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def graph_panel(ax, edges):
    positions = {
        "CD": (0.08, 0.50), "MI": (0.35, 0.50), "HYP": (0.68, 0.50),
        "NORM": (0.515, 0.88), "STTC": (0.515, 0.12),
    }
    colors = {"NORM": "#dcefd4", "MI": "#f7d9e5", "STTC": "#ddd7f0", "CD": "#d8e4f6", "HYP": "#fae5bf"}
    borders = {"NORM": "#69ad50", "MI": "#ec6896", "STTC": "#8875cb", "CD": "#729ddc", "HYP": "#ee9b2f"}
    for edge in edges:
        u, v = edge["class_u"], edge["class_v"]
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], color="#5c78a8", lw=2.2, alpha=0.75, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1e-8)
        sign = 1.0 if int(edge["rank"]) % 2 else -1.0
        mx += sign * 0.040 * (-dy / length)
        my += sign * 0.040 * (dx / length)
        ax.text(mx, my, f"{edge['fused_score']:.3f}", ha="center", va="center",
                fontsize=8.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2}, zorder=3)
    for name, (x, y) in positions.items():
        ax.scatter([x], [y], s=2100, color=colors[name], edgecolor=borders[name], linewidth=1.3, zorder=2)
        ax.text(x, y, name, ha="center", va="center", fontsize=10, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(c) Top-6 fused relations", pad=10, fontsize=11)


def vector_colorbar(ax, cmap):
    """Draw a vector-only horizontal color scale below a heatmap axis."""
    cax = ax.inset_axes([0.0, -0.15, 1.0, 0.05])
    n_bins = 64
    for i in range(n_bins):
        x0 = i / n_bins
        cax.add_patch(Rectangle(
            (x0, 0.0),
            1.0 / n_bins,
            1.0,
            facecolor=cmap((i + 0.5) / n_bins),
            edgecolor="none",
        ))
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(0.0, 1.0)
    cax.set_yticks([])
    cax.set_xticks(np.linspace(0.0, 1.0, 6))
    cax.tick_params(axis="x", labelsize=8, pad=2)
    for spine in cax.spines.values():
        spine.set_linewidth(0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-labels", type=Path, required=True)
    ap.add_argument("--oof-probs", type=Path, required=True)
    ap.add_argument("--oof-targets", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--prior-reference-threshold", type=float, default=0.10)
    ap.add_argument("--source-description", required=True,
                    help="Exact seed/run identifier; prevents unlabeled illustrative output.")
    args = ap.parse_args()

    y_train = np.load(args.train_labels)
    probs = np.load(args.oof_probs)
    targets = np.load(args.oof_targets)
    prior = prior_from_labels(y_train)
    raw = confusion_raw_from_oof(probs, targets)
    normalized, raw_max = normalize_confusion(raw)
    fused = 0.5 * prior + 0.5 * normalized
    np.fill_diagonal(fused, 1.0)
    edges = top_edges(prior, raw, normalized, fused, args.k)
    retained = [row for row in edges if row["retained"]]

    tri = np.triu_indices(len(CLASS_NAMES), k=1)
    reference_budget = int(np.sum(prior[tri] > args.prior_reference_threshold))
    if reference_budget != args.k:
        raise ValueError(
            f"Prior threshold gives K={reference_budget}, not requested K={args.k}; "
            "do not use the threshold-derived-budget wording."
        )
    if not np.isclose(normalized[tri].max(), 1.0):
        raise AssertionError("Normalized confusion maximum must be 1.0")
    for row in retained:
        expected = 0.5 * row["prior_score"] + 0.5 * row["confusion_normalized_score"]
        if not np.isclose(expected, row["fused_score"]):
            raise AssertionError("Fused edge score mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_csv(prior, args.out_dir / "prior_matrix.csv")
    matrix_csv(raw, args.out_dir / "confusion_raw_matrix.csv")
    matrix_csv(normalized, args.out_dir / "confusion_normalized_matrix.csv")
    matrix_csv(fused, args.out_dir / "fused_matrix.csv")
    pd.DataFrame(retained).drop(columns=["i", "j", "retained"]).to_csv(
        args.out_dir / "top6_edges.csv", index=False
    )
    metadata = {
        "source_description": args.source_description,
        "class_order": CLASS_NAMES,
        "n_train_records": int(len(y_train)),
        "n_oof_records": int(len(targets)),
        "raw_confusion_max_offdiagonal": raw_max,
        "normalized_confusion_max_offdiagonal": float(normalized[tri].max()),
        "prior_reference_threshold": args.prior_reference_threshold,
        "reference_edge_budget": reference_budget,
        "retention_rule": f"top-{args.k} undirected off-diagonal fused scores",
        "fusion": "0.5 * S_prior + 0.5 * S_conf_normalized",
    }
    (args.out_dir / "figure3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    blue = LinearSegmentedColormap.from_list("prior", ["#f7f9fc", "#77a5dd"])
    rose = LinearSegmentedColormap.from_list("conf", ["#fff8fa", "#e66d9d"])
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.18]})
    im0 = heatmap(axes[0], prior, blue, r"(a) $\mathbf{S}_{\mathrm{prior}}$", decimals=4)
    im1 = heatmap(axes[1], normalized, rose, r"(b) Normalized $\mathbf{S}_{\mathrm{conf}}$", decimals=2)
    graph_panel(axes[2], retained)
    vector_colorbar(axes[0], blue)
    vector_colorbar(axes[1], rose)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.12, wspace=0.34)
    fig.savefig(args.out_dir / "relation_graph_figure_final.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "relation_graph_figure_final.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
