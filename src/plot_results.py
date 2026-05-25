"""
plot_results.py

Generates all result figures and comparison tables for the report:

  1. Privacy-utility tradeoff scatter plot
       X: AI4Privacy micro F1  (general-domain utility — want high)
       Y: Clinical overlap recall  (clinical PII sensitivity — want high)
       Points: baseline + 3 LoRA configs, annotated with train size

  2. Evasion rate comparison bar chart
       Grouped bars: baseline vs LoRA 20k for each attack
       Colour-coded: character-level attacks (red) vs domain-level attacks (blue)

  3. Catastrophic forgetting curves
       AI4Privacy micro F1 vs epoch for all 3 LoRA configs

  4. Prints a plain-text comparison table (stdout + saved as JSON)

Usage:
    python src/plot_results.py
    python src/plot_results.py --output-dir results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (no display required on Hyak)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIGS = {
    "baseline":    {"label": "Baseline",   "train_size": None, "color": "#444444", "marker": "s"},
    "lora_r8_n1000":  {"label": "LoRA 1k",  "train_size": 1000,  "color": "#2196F3", "marker": "o"},
    "lora_r8_n5000":  {"label": "LoRA 5k",  "train_size": 5000,  "color": "#4CAF50", "marker": "o"},
    "lora_r8_n20000": {"label": "LoRA 20k", "train_size": 20000, "color": "#F44336", "marker": "o"},
}

CHAR_LEVEL_ATTACKS   = {"homoglyph", "zwsp", "whitespace"}
DOMAIN_LEVEL_ATTACKS = {"abbreviation", "honorific", "field_format"}
ATTACK_ORDER = ["homoglyph", "zwsp", "whitespace", "abbreviation", "honorific", "field_format"]

BASELINE_AI4P_F1 = None  # loaded from results/baseline_eval.json


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def get_attack_dict(data: dict) -> dict:
    """Handle both old format (attack keys at top level) and new format ({run_label, attacks})."""
    if data is None:
        return {}
    if "attacks" in data:
        return data["attacks"]
    return data  # old baseline format


def weighted_evasion_rate(attack_data: dict) -> float | None:
    """Compute weighted-average evasion rate across all entity types for one attack."""
    total_det = total_evd = 0
    for etype, c in attack_data.items():
        od = c.get("originally_detected", 0)
        ev = c.get("evaded", 0)
        total_det += od
        total_evd += ev
    if total_det == 0:
        return None
    return total_evd / total_det


def load_clinical_f1(name: str) -> dict:
    """Load clinical overlap micro F1, precision, recall for one config."""
    path = f"results/clinical_eval_{name}.json"
    data = load_json(path)
    if data is None:
        return {}
    ov = data["overlap_f1"]["micro avg"]
    return {"precision": ov["precision"], "recall": ov["recall"], "f1": ov["f1"]}


def load_ai4p_f1(name: str) -> float | None:
    """
    Load best AI4Privacy micro F1 for one config.
    Baseline: from results/baseline_eval.json
    LoRA:     last epoch of results/checkpoints/<name>/forgetting_history.json
    """
    if name == "baseline":
        data = load_json("results/baseline_eval.json")
        if data is None:
            return None
        return data["ai4privacy"]["micro avg"]["f1-score"]
    else:
        data = load_json(f"results/checkpoints/{name}/forgetting_history.json")
        if data is None:
            return None
        # Return the best (max) epoch F1 — forgetting plateaus so last ≈ best
        return max(e["ai4privacy_micro_f1"] for e in data)


def load_attack_evasion(name: str) -> dict:
    """
    Load weighted evasion rates per attack for one config.
    Returns {attack_name: evasion_rate_or_None}
    """
    if name == "baseline":
        path = "results/attack_eval.json"
    else:
        path = f"results/attack_eval_{name}.json"
    data = load_json(path)
    attacks = get_attack_dict(data)
    rates = {}
    for atk in ATTACK_ORDER:
        if atk in attacks:
            rates[atk] = weighted_evasion_rate(attacks[atk])
        else:
            rates[atk] = None
    return rates


# ---------------------------------------------------------------------------
# Plot 1: Privacy-Utility Tradeoff
# ---------------------------------------------------------------------------

def plot_privacy_utility(configs_data: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, cfg in CONFIGS.items():
        d = configs_data[name]
        x = d.get("ai4p_f1")
        y = d.get("clinical_recall")
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=120, color=cfg["color"], marker=cfg["marker"],
                   zorder=5, label=cfg["label"])
        ax.annotate(cfg["label"], (x, y),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=cfg["color"])

    # Ideal point arrow
    ax.annotate("", xy=(0.98, 0.98), xytext=(0.85, 0.85),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.text(0.87, 0.86, "ideal", transform=ax.transAxes,
            fontsize=8, color="gray", ha="left")

    ax.set_xlabel("AI4Privacy micro F1  (general-domain utility)", fontsize=11)
    ax.set_ylabel("Clinical overlap recall  (PII sensitivity)", fontsize=11)
    ax.set_title("Privacy–Utility Tradeoff: LoRA Fine-Tuning", fontsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim(0.9, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = output_dir / "privacy_utility_tradeoff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Evasion Rate Comparison (baseline vs LoRA 20k)
# ---------------------------------------------------------------------------

def plot_evasion_comparison(configs_data: dict, output_dir: Path):
    baseline_rates = configs_data["baseline"].get("evasion", {})
    lora20k_rates  = configs_data["lora_r8_n20000"].get("evasion", {})

    # Only attacks where at least baseline has a rate
    attacks = [a for a in ATTACK_ORDER if baseline_rates.get(a) is not None]
    if not attacks:
        print("No evasion data available yet — skipping evasion plot.")
        return

    x = np.arange(len(attacks))
    width = 0.35

    base_vals = [baseline_rates.get(a) or 0.0 for a in attacks]
    lora_vals  = [lora20k_rates.get(a)  if lora20k_rates.get(a) is not None else float("nan")
                  for a in attacks]

    # Colors per bar: red = char-level, blue = domain-level
    bar_colors = ["#E53935" if a in CHAR_LEVEL_ATTACKS else "#1E88E5" for a in attacks]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_base = ax.bar(x - width/2, base_vals, width, label="Baseline",
                       color=bar_colors, alpha=0.9, edgecolor="white")
    bars_lora = ax.bar(x + width/2, lora_vals, width, label="LoRA 20k",
                       color=bar_colors, alpha=0.4, edgecolor=bar_colors, linewidth=1.5,
                       hatch="//")

    # Value labels
    for bar in bars_base:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=8)
    for bar in bars_lora:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=10)
    ax.set_ylabel("Weighted evasion rate", fontsize=11)
    ax.set_title("Adversarial Evasion Rates: Baseline vs LoRA 20k", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.axhline(0, color="black", linewidth=0.5)

    # Legend entries
    char_patch   = mpatches.Patch(color="#E53935", label="Character-level attack")
    domain_patch = mpatches.Patch(color="#1E88E5", label="Domain-level attack")
    base_patch   = mpatches.Patch(facecolor="gray", alpha=0.9, label="Baseline")
    lora_patch   = mpatches.Patch(facecolor="gray", alpha=0.4, hatch="//", label="LoRA 20k")
    ax.legend(handles=[char_patch, domain_patch, base_patch, lora_patch],
              fontsize=9, loc="upper right")

    # Bracket to label char-level vs domain-level groups
    n_char = sum(1 for a in attacks if a in CHAR_LEVEL_ATTACKS)
    if n_char > 0 and n_char < len(attacks):
        mid_char   = (n_char - 1) / 2
        mid_domain = n_char + (len(attacks) - n_char - 1) / 2
        ax.annotate("char-level", xy=(mid_char, -0.12), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=9, color="#E53935",
                    arrowprops=None)
        ax.annotate("domain-level", xy=(mid_domain, -0.12), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=9, color="#1E88E5")

    fig.tight_layout()
    out = output_dir / "evasion_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Catastrophic Forgetting Curves
# ---------------------------------------------------------------------------

def plot_forgetting_curves(output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))

    plotted = False
    for name, cfg in CONFIGS.items():
        if name == "baseline":
            continue
        data = load_json(f"results/checkpoints/{name}/forgetting_history.json")
        if data is None:
            continue
        epochs = [e["epoch"] for e in data]
        f1s    = [e["ai4privacy_micro_f1"] for e in data]
        ax.plot(epochs, f1s, marker="o", color=cfg["color"],
                label=cfg["label"], linewidth=2, markersize=6)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("No forgetting history available — skipping forgetting plot.")
        return

    # Baseline reference line
    base_f1 = load_ai4p_f1("baseline")
    if base_f1 is not None:
        ax.axhline(base_f1, linestyle="--", color="#444444", linewidth=1.2,
                   label=f"Baseline ({base_f1:.3f})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("AI4Privacy micro F1", fontsize=11)
    ax.set_title("Catastrophic Forgetting During LoRA Fine-Tuning", fontsize=12)
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "forgetting_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Table: evasion comparison across all configs
# ---------------------------------------------------------------------------

def print_evasion_table(configs_data: dict):
    names  = list(CONFIGS.keys())
    labels = [CONFIGS[n]["label"] for n in names]

    print(f"\n{'='*80}")
    print("  EVASION RATE COMPARISON  (weighted avg across entity types)")
    print(f"{'='*80}")
    header = f"  {'Attack':<16} {'Type':<8}" + "".join(f"  {l:>10}" for l in labels)
    print(header)
    print(f"  {'-'*74}")

    for atk in ATTACK_ORDER:
        atype = "char" if atk in CHAR_LEVEL_ATTACKS else "domain"
        row = f"  {atk:<16} {atype:<8}"
        for name in names:
            rate = configs_data[name].get("evasion", {}).get(atk)
            if rate is None:
                row += f"  {'N/A':>10}"
            else:
                row += f"  {rate:>9.1%} "
        print(row)

    print(f"\n{'='*80}")
    print("  CLINICAL SPAN F1  (test set: 1000 MIMIC notes, overlap matching)")
    print(f"{'='*80}")
    print(f"  {'Metric':<16}" + "".join(f"  {l:>10}" for l in labels))
    print(f"  {'-'*58}")
    for metric in ["precision", "recall", "f1"]:
        row = f"  {metric:<16}"
        for name in names:
            v = configs_data[name].get("clinical_" + metric)
            row += f"  {v:>10.3f}" if v is not None else f"  {'N/A':>10}"
        print(row)

    print(f"\n{'='*80}")
    print("  AI4PRIVACY MICRO F1  (general-domain utility)")
    print(f"{'='*80}")
    row = f"  {'AI4P F1':<16}"
    for name in names:
        v = configs_data[name].get("ai4p_f1")
        row += f"  {v:>10.3f}" if v is not None else f"  {'N/A':>10}"
    print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/figures",
                        help="Directory for output figures and JSON summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all data ────────────────────────────────────────────────────────
    print("Loading results...")
    configs_data = {}
    for name in CONFIGS:
        clinical = load_clinical_f1(name)
        configs_data[name] = {
            "ai4p_f1":         load_ai4p_f1(name),
            "clinical_precision": clinical.get("precision"),
            "clinical_recall":    clinical.get("recall"),
            "clinical_f1":        clinical.get("f1"),
            "evasion":         load_attack_evasion(name),
        }

    # ── Print comparison table ───────────────────────────────────────────────
    print_evasion_table(configs_data)

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(configs_data, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_privacy_utility(configs_data, output_dir)
    plot_evasion_comparison(configs_data, output_dir)
    plot_forgetting_curves(output_dir)

    print("\nAll done. Figures in:", output_dir)


if __name__ == "__main__":
    main()
