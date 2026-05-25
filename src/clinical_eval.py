"""
clinical_eval.py

Evaluate openai/privacy-filter (baseline or LoRA-adapted) on the PII-injected
MIMIC test set, computing span-level F1 with both exact and overlap matching.

Usage — baseline model:
    python src/clinical_eval.py \
        --data data/mimic_pii_injected.jsonl \
        --output results/clinical_eval_baseline.json

Usage — LoRA adapter:
    python src/clinical_eval.py \
        --adapter results/checkpoints/lora_r8_n5000/adapter \
        --output results/clinical_eval_lora_r8_n5000.json

Span matching modes:
  exact   — predicted (start, end, type) must exactly match gold span
  overlap — predicted span overlaps gold span with same entity type
            (fairer for subword tokenization misalignments)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import pipeline


MODEL_ID = "openai/privacy-filter"

# Fixed test split — never used during LoRA training
TEST_START = 21000
TEST_END   = 22000


# ---------------------------------------------------------------------------
# Span F1 — exact matching
# ---------------------------------------------------------------------------

def span_f1_exact(gold_spans, pred_spans):
    """
    gold_spans / pred_spans: list of (start, end, entity_type)
    TP: predicted span matches gold on all three fields exactly.
    Returns per-type + micro avg report dict.
    """
    gold_by_type = defaultdict(set)
    pred_by_type = defaultdict(set)
    for s, e, t in gold_spans:
        gold_by_type[t].add((s, e))
    for s, e, t in pred_spans:
        pred_by_type[t].add((s, e))

    return _compute_report(gold_by_type, pred_by_type)


# ---------------------------------------------------------------------------
# Span F1 — overlap matching
# ---------------------------------------------------------------------------

def span_f1_overlap(gold_spans, pred_spans):
    """
    TP: a predicted span overlaps a gold span of the same type.
    Each gold span can be matched at most once (greedy left-to-right).
    """
    gold_by_type  = defaultdict(list)
    pred_by_type  = defaultdict(list)
    for s, e, t in gold_spans:
        gold_by_type[t].append((s, e))
    for s, e, t in pred_spans:
        pred_by_type[t].append((s, e))

    gold_by_type_count = defaultdict(set)
    pred_by_type_count = defaultdict(set)
    matched_gold = defaultdict(set)

    for t in set(list(gold_by_type.keys()) + list(pred_by_type.keys())):
        golds = gold_by_type[t]
        preds = pred_by_type[t]
        for gi, (gs, ge) in enumerate(golds):
            gold_by_type_count[t].add(gi)
        for pi, (ps, pe) in enumerate(preds):
            pred_by_type_count[t].add(pi)
            # Try to match to an unmatched gold span
            for gi, (gs, ge) in enumerate(golds):
                if gi in matched_gold[t]:
                    continue
                if ps < ge and gs < pe:   # overlap
                    matched_gold[t].add(gi)
                    break

    # Build TP/FP/FN counts per type
    report = {}
    total_tp = total_fp = total_fn = 0
    all_types = set(gold_by_type_count) | set(pred_by_type_count)
    for t in all_types:
        tp = len(matched_gold[t])
        fp = len(pred_by_type_count[t]) - tp
        fn = len(gold_by_type_count[t]) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report[t] = {"precision": prec, "recall": rec, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn,
                     "support": len(gold_by_type_count[t])}
        total_tp += tp; total_fp += fp; total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    report["micro avg"] = {"precision": micro_p, "recall": micro_r, "f1": micro_f,
                           "tp": total_tp, "fp": total_fp, "fn": total_fn,
                           "support": sum(len(v) for v in gold_by_type_count.values())}
    return report


def _compute_report(gold_by_type, pred_by_type):
    all_types = set(gold_by_type) | set(pred_by_type)
    report = {}
    total_tp = total_fp = total_fn = 0
    for t in all_types:
        g, p = gold_by_type[t], pred_by_type[t]
        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report[t] = {"precision": prec, "recall": rec, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn,
                     "support": len(g)}
        total_tp += tp; total_fp += fp; total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    report["micro avg"] = {"precision": micro_p, "recall": micro_r, "f1": micro_f,
                           "tp": total_tp, "fp": total_fp, "fn": total_fn,
                           "support": sum(len(v) for v in gold_by_type.values())}
    return report


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_report(report, mode, label):
    print(f"\n{'='*60}")
    print(f"  {label}  [{mode} matching]")
    print(f"{'='*60}")
    print(f"  {'Entity':<22} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print(f"  {'-'*48}")
    for t, v in sorted(report.items()):
        if t == "micro avg":
            continue
        print(f"  {t:<22} {v['precision']:>6.3f} {v['recall']:>6.3f} "
              f"{v['f1']:>6.3f} {v['support']:>6}")
    print(f"  {'-'*48}")
    v = report["micro avg"]
    print(f"  {'micro avg':<22} {v['precision']:>6.3f} {v['recall']:>6.3f} "
          f"{v['f1']:>6.3f} {v['support']:>6}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mimic_pii_injected.jsonl")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter dir. If omitted, uses base model.")
    parser.add_argument("--output", default="results/clinical_eval.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # ── Load test split ──────────────────────────────────────────────────────
    print(f"Loading test split ({TEST_START}–{TEST_END}) from {args.data}...")
    with open(args.data) as f:
        all_records = [json.loads(l) for l in f]

    if len(all_records) < TEST_END:
        raise ValueError(f"Need {TEST_END} records, got {len(all_records)}")

    test_records = all_records[TEST_START:TEST_END]
    print(f"Test set: {len(test_records)} notes")

    # ── Load model ───────────────────────────────────────────────────────────
    if args.adapter:
        print(f"Loading LoRA adapter from {args.adapter}...")
        from peft import PeftModel
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        base = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
        model = PeftModel.from_pretrained(base, args.adapter)
        model = model.merge_and_unload()  # merge LoRA weights for fast inference
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=args.device,
        )
        run_label = f"LoRA ({Path(args.adapter).parent.name})"
    else:
        print(f"Loading base model {MODEL_ID}...")
        pipe = pipeline(
            "token-classification",
            model=MODEL_ID,
            aggregation_strategy="simple",
            device=args.device,
        )
        run_label = "Baseline (no fine-tuning)"

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"Running inference on {len(test_records)} notes...")
    texts = [r["text"] for r in test_records]

    all_preds = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i: i + args.batch_size]
        preds = pipe(batch)
        all_preds.extend(preds)
        if (i // args.batch_size + 1) % 10 == 0:
            print(f"  {i + len(batch)}/{len(texts)} notes done...")

    # ── Compute F1 ───────────────────────────────────────────────────────────
    gold_spans_all, pred_spans_all = [], []
    for record, preds in zip(test_records, all_preds):
        for e in record["entities"]:
            gold_spans_all.append((e["start"], e["end"], e["entity_type"]))
        for p in preds:
            pred_spans_all.append((p["start"], p["end"], p["entity_group"]))

    exact_report   = span_f1_exact(gold_spans_all, pred_spans_all)
    overlap_report = span_f1_overlap(gold_spans_all, pred_spans_all)

    print_report(exact_report,   "exact",   run_label)
    print_report(overlap_report, "overlap", run_label)

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "run_label":      run_label,
        "adapter":        args.adapter,
        "n_test_notes":   len(test_records),
        "exact_f1":       exact_report,
        "overlap_f1":     overlap_report,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
