"""
eval_forgetting.py

Post-hoc evaluation of saved LoRA adapters on AI4Privacy to produce
corrected AI4Privacy F1 values.

Context: the ForgettingCallback in lora_finetune.py wrote forgetting_history.json
during training using a buggy label map (missing GIVENNAME/SURNAME/DATEOFBIRTH/etc.).
This script re-evaluates each saved adapter with the fixed label map from the
updated baseline_eval.py, without requiring any retraining.

Output per adapter:
    results/checkpoints/<name>/forgetting_history_corrected.json
    {
        "name": "lora_r8_n5000",
        "ai4privacy_micro_f1_corrected": 0.XXXX,
        "original_history": [...]   # original per-epoch values for reference
    }

Usage:
    python src/eval_forgetting.py --device 0 --max-samples 2000
"""

import argparse
import json
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# evaluate_on_ai4privacy now uses the corrected label map and span_f1
from baseline_eval import evaluate_on_ai4privacy

MODEL_ID = "openai/privacy-filter"

ADAPTERS = {
    "lora_r8_n1000":  "results/checkpoints/lora_r8_n1000/adapter",
    "lora_r8_n5000":  "results/checkpoints/lora_r8_n5000/adapter",
    "lora_r8_n20000": "results/checkpoints/lora_r8_n20000/adapter",
}


def eval_adapter(adapter_path: str, device: int, max_samples: int) -> float:
    """Load a saved LoRA adapter, merge weights, evaluate on AI4Privacy."""
    print(f"  Loading adapter: {adapter_path}")
    base = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )
    report = evaluate_on_ai4privacy(pipe, max_samples=max_samples)
    micro = report.get("micro avg", {})
    f1 = micro.get("f1-score", 0.0)
    prec = micro.get("precision", 0.0)
    rec = micro.get("recall", 0.0)
    print(f"  AI4Privacy micro — P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    # Also print per-type F1 for full picture
    for label, vals in sorted(report.items()):
        if label == "micro avg" or not isinstance(vals, dict):
            continue
        print(f"    {label:<22} P={vals['precision']:.3f}  R={vals['recall']:.3f}"
              f"  F1={vals['f1-score']:.3f}  N={int(vals['support'])}")
    return f1, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="AI4Privacy examples (match the 2000 used during training)")
    parser.add_argument("--adapters", nargs="+", default=list(ADAPTERS.keys()),
                        choices=list(ADAPTERS.keys()),
                        help="Which adapters to evaluate (default: all three)")
    args = parser.parse_args()

    # Also evaluate the baseline (no adapter) for a corrected reference point
    print(f"\n{'='*55}")
    print("  Baseline (no adapter)")
    print(f"{'='*55}")
    base_pipe = pipeline(
        "token-classification",
        model=MODEL_ID,
        aggregation_strategy="simple",
        device=args.device,
    )
    base_report = evaluate_on_ai4privacy(base_pipe, max_samples=args.max_samples)
    base_micro = base_report.get("micro avg", {})
    base_f1 = base_micro.get("f1-score", 0.0)
    print(f"  Baseline AI4Privacy micro F1 (corrected): {base_f1:.4f}")
    for label, vals in sorted(base_report.items()):
        if label == "micro avg" or not isinstance(vals, dict):
            continue
        print(f"    {label:<22} P={vals['precision']:.3f}  R={vals['recall']:.3f}"
              f"  F1={vals['f1-score']:.3f}  N={int(vals['support'])}")

    results_summary = {"baseline": {"ai4privacy_micro_f1_corrected": base_f1,
                                    "full_report": base_report}}

    # Evaluate each adapter
    for name in args.adapters:
        adapter_path = ADAPTERS[name]
        adapter_dir = Path(adapter_path)

        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        if not adapter_dir.exists():
            print(f"  [SKIP] adapter not found at {adapter_path}")
            continue

        f1, full_report = eval_adapter(adapter_path, args.device, args.max_samples)

        # Load original forgetting history for reference
        hist_path = adapter_dir.parent / "forgetting_history.json"
        orig_history = []
        if hist_path.exists():
            with open(hist_path) as fh:
                orig_history = json.load(fh)
            orig_f1s = [e["ai4privacy_micro_f1"] for e in orig_history]
            print(f"  Original (buggy) per-epoch F1: {orig_f1s}")
            print(f"  Original final F1:   {orig_f1s[-1]:.4f}  →  Corrected: {f1:.4f}")

        # Write corrected file
        corrected = {
            "name": name,
            "ai4privacy_micro_f1_corrected": f1,
            "full_report_corrected": full_report,
            "original_history": orig_history,
        }
        out_path = adapter_dir.parent / "forgetting_history_corrected.json"
        with open(out_path, "w") as fh:
            json.dump(corrected, fh, indent=2)
        print(f"  Saved: {out_path}")

        results_summary[name] = {"ai4privacy_micro_f1_corrected": f1}

    # Print comparison table
    print(f"\n{'='*55}")
    print("  SUMMARY — Corrected AI4Privacy micro F1")
    print(f"{'='*55}")
    print(f"  {'Config':<20} {'F1 (corrected)':>16}")
    print(f"  {'-'*38}")
    for name, d in results_summary.items():
        print(f"  {name:<20} {d['ai4privacy_micro_f1_corrected']:>16.4f}")
    print()


if __name__ == "__main__":
    main()
