"""
plot_improved.py

Redraws two figures with corrected data and better readability:
  1. privacy_utility_tradeoff.png  — uses clinical F1 (not recall); axes tight around data
  2. forgetting_curves.png         — uses corrected AI4P F1; before/after design instead
                                     of buggy per-epoch training curves

Usage:
    python src/plot_improved.py
    python src/plot_improved.py --output-dir results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

CONFIGS = {
    "baseline":       {"label": "Baseline",  "train_size": 0,     "color": "#444444", "marker": "s"},
    "lora_r8_n1000":  {"label": "LoRA  1k",  "train_size": 1000,  "color": "#2196F3", "marker": "o"},
    "lora_r8_n5000":  {"label": "LoRA  5k",  "train_size": 5000,  "color": "#4CAF50", "marker": "o"},
    "lora_r8_n20000": {"label": "LoRA 20k",  "train_size": 20000, "color": "#F44336", "marker": "o"},
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


def load_all(base="results"):
    data = {}
    for name in CONFIGS:
        # Clinical F1 (corrected overlap matching)
        cf = load_json(f"{base}/clinical_eval_{name}.json")
        if cf:
            ov = cf["overlap_f1"]["micro avg"]
            clinical = {"precision": ov["precision"], "recall": ov["recall"], "f1": ov["f1"]}
        else:
            clinical = {}

        # AI4Privacy F1 (corrected label map + overlap matching)
        if name == "baseline":
            bf = load_json(f"{base}/baseline_eval.json")
            ai4p_f1 = bf["ai4privacy"]["micro avg"]["f1-score"] if bf else None
        else:
            corr = load_json(f"{base}/checkpoints/{name}/forgetting_history_corrected.json")
            ai4p_f1 = corr.get("ai4privacy_micro_f1_corrected") if corr else None

        data[name] = {
            "ai4p_f1":         ai4p_f1,
            "clinical_f1":     clinical.get("f1"),
            "clinical_recall": clinical.get("recall"),
            "clinical_precision": clinical.get("precision"),
        }
    return data


# ── Plot 1: Privacy-Utility Tradeoff ─────────────────────────────────────────

def plot_privacy_utility(data, output_dir):
    """
    X: AI4Privacy micro F1   (general-domain utility — higher is better)
    Y: Clinical overlap F1   (clinical PII detection — higher is better)

    Design decisions:
    - Axes zoomed tightly around the data (total span ~0.04 on X, ~0.09 on Y)
    - Green shaded quadrant above+right of baseline shows "improvement" region
    - Arrow from baseline to LoRA cluster centroid to make the story explicit
    - Text label boxes (white bg) so nothing overlaps the points
    - LoRA 5k/20k sit almost on top of each other — staggered labels handle it
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    # ── Data ────────────────────────────────────────────────────────────────
    xs, ys = {}, {}
    for name in CONFIGS:
        x = data[name].get("ai4p_f1")
        y = data[name].get("clinical_f1")
        if x is not None and y is not None:
            xs[name], ys[name] = x, y

    bx, by = xs["baseline"], ys["baseline"]

    # ── Tight axis limits — data fills ~70% of each axis ────────────────────
    x_lo, x_hi = 0.382, 0.444
    y_lo, y_hi = 0.574, 0.686
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # ── "Improvement over baseline" shaded quadrant ──────────────────────────
    ax.fill_between(
        [bx, x_hi], [by, by], [y_hi, y_hi],
        color="#4CAF50", alpha=0.07, zorder=0,
    )
    ax.text(x_hi - 0.001, y_hi - 0.002,
            "better than\nbaseline",
            ha="right", va="top", fontsize=9, color="#388E3C",
            style="italic", transform=ax.transData)

    # ── Vertical/horizontal reference lines through baseline ─────────────────
    ax.axvline(bx, color="#aaaaaa", linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(by, color="#aaaaaa", linewidth=0.8, linestyle=":", zorder=1)

    # ── LoRA ablation curve (20k → 5k → 1k, ordered by train size) ──────────
    lora_order = ["lora_r8_n20000", "lora_r8_n5000", "lora_r8_n1000"]
    lx = [xs[n] for n in lora_order if n in xs]
    ly = [ys[n] for n in lora_order if n in ys]
    if lx:
        ax.plot(lx, ly, color="#bbbbbb", linewidth=1.5, linestyle="--",
                zorder=2, label="_nolegend_")

    # ── Story arrow: baseline → centroid of LoRA cluster ────────────────────
    cx = sum(xs[n] for n in lora_order if n in xs) / len(lx)
    cy = sum(ys[n] for n in lora_order if n in ys) / len(ly)
    ax.annotate(
        "", xy=(cx, cy), xytext=(bx, by),
        arrowprops=dict(
            arrowstyle="-|>", color="#2e7d32",
            lw=2.0, mutation_scale=18,
            connectionstyle="arc3,rad=0.18",
        ),
        zorder=4,
    )
    # Label the arrow mid-point
    ax.text((bx + cx) / 2 - 0.003, (by + cy) / 2 - 0.008,
            "LoRA fine-\ntuning ↑", ha="right", va="top",
            fontsize=9, color="#2e7d32", style="italic")

    # ── Scatter points ───────────────────────────────────────────────────────
    for name, cfg in CONFIGS.items():
        if name not in xs:
            continue
        ax.scatter(xs[name], ys[name],
                   s=260, color=cfg["color"], marker=cfg["marker"],
                   zorder=6, edgecolors="white", linewidths=2.0)

    # ── Annotations with white bounding boxes ────────────────────────────────
    # Hand-tuned (x_offset_pts, y_offset_pts) so no label covers a point
    label_cfg = {
        "baseline":       ("Baseline\n(no fine-tuning)", -8, -32, "right"),
        "lora_r8_n1000":  ("LoRA 1k",                   12,   8, "left"),
        "lora_r8_n5000":  ("LoRA 5k",                   12,   16, "left"),
        "lora_r8_n20000": ("LoRA 20k",                  -8,  10, "right"),
    }
    bbox_style = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85)
    for name, cfg in CONFIGS.items():
        if name not in xs:
            continue
        txt, ox, oy, ha = label_cfg[name]
        ax.annotate(
            txt, (xs[name], ys[name]),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=11, color=cfg["color"], fontweight="bold",
            ha=ha, va="center",
            bbox=bbox_style,
            arrowprops=dict(arrowstyle="-", color=cfg["color"],
                            lw=1.0, shrinkA=0, shrinkB=8),
            zorder=7,
        )

    # ── Axes and title ────────────────────────────────────────────────────────
    ax.set_xlabel("AI4Privacy Micro F1  (general-domain utility ↑)", fontsize=12)
    ax.set_ylabel("Clinical Overlap F1  (clinical PII detection ↑)", fontsize=12)
    ax.set_title("Privacy–Utility Tradeoff Across LoRA Configurations",
                 fontsize=13, fontweight="bold", pad=14)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)

    fig.tight_layout()
    out = output_dir / "privacy_utility_tradeoff.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 2: Utility Before vs After LoRA ─────────────────────────────────────

def plot_forgetting_curves(data, output_dir):
    """
    Shows AI4Privacy micro F1 as a function of LoRA training size.
    Baseline = pre-training value (no fine-tuning).
    LoRA values = corrected final-checkpoint F1.

    Replaces the old per-epoch plot which used buggy values.
    Key message: LoRA preserves (or slightly improves) general-domain utility.
    """
    sizes   = [0,    1000,            5000,            20000           ]
    names   = ["baseline", "lora_r8_n1000", "lora_r8_n5000", "lora_r8_n20000"]
    colors  = [CONFIGS[n]["color"] for n in names]
    labels  = ["Baseline\n(no fine-tuning)", "LoRA 1k", "LoRA 5k", "LoRA 20k"]

    f1s = [data[n].get("ai4p_f1") for n in names]
    if any(v is None for v in f1s):
        print("Missing AI4P F1 data — skipping forgetting plot.")
        return

    x_idx = np.arange(len(names))   # categorical positions

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Shaded region: improvement over baseline
    baseline_f1 = f1s[0]
    for i in range(1, len(names)):
        ax.fill_between(
            [x_idx[i] - 0.25, x_idx[i] + 0.25],
            baseline_f1, f1s[i],
            color="#4CAF50" if f1s[i] >= baseline_f1 else "#F44336",
            alpha=0.12, zorder=1,
        )

    # Main line
    ax.plot(x_idx, f1s, marker="o", color="#333333",
            linewidth=2.2, markersize=0, zorder=3)

    # Colored dots per config
    for i, (xi, v, c) in enumerate(zip(x_idx, f1s, colors)):
        ax.scatter([xi], [v], s=160, color=c, zorder=5,
                   edgecolors="white", linewidths=1.5)

    # Value labels above each point
    for xi, v in zip(x_idx, f1s):
        ax.text(xi, v + 0.0025, f"{v:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222222")

    # Delta annotations (change from baseline) for LoRA configs
    for i in range(1, len(names)):
        delta = f1s[i] - baseline_f1
        sign  = "+" if delta >= 0 else ""
        mid_y = (baseline_f1 + f1s[i]) / 2
        ax.text(x_idx[i] + 0.28, mid_y, f"({sign}{delta:.3f})",
                ha="left", va="center", fontsize=9,
                color="#2e7d32" if delta >= 0 else "#c62828")

    # Baseline dashed reference line
    ax.axhline(baseline_f1, color="#444444", linewidth=1.2,
               linestyle=":", alpha=0.6, zorder=2)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("AI4Privacy Micro F1", fontsize=12)

    y_lo = min(f1s) - 0.025
    y_hi = max(f1s) + 0.025
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.tick_params(axis="x", length=0, labelsize=11)
    ax.tick_params(axis="y", labelsize=10)

    ax.set_title("General-Domain Utility vs LoRA Training Size\n"
                 "(Minimal Catastrophic Forgetting — Corrected Evaluation)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.7)

    # Legend chips
    patch_good = mpatches.Patch(color="#4CAF50", alpha=0.3, label="Improvement over baseline")
    ax.legend(handles=[patch_good], fontsize=9, loc="lower right")

    fig.tight_layout()
    out = output_dir / "forgetting_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_all(args.results_dir)

    print("\nData loaded:")
    for name, d in data.items():
        print(f"  {name:<20}  ai4p_f1={d['ai4p_f1']}  clinical_f1={d['clinical_f1']}")

    plot_privacy_utility(data, output_dir)
    plot_forgetting_curves(data, output_dir)
    print("\nDone. Figures in:", output_dir)


if __name__ == "__main__":
    main()
