"""
Baseline evaluation of openai/privacy-filter on general (AI4Privacy) and
clinical (MIMIC) domains. Outputs per-entity-type F1 and an overall F1 table.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import pipeline


# Map AI4Privacy fine-grained labels to privacy-filter's entity types.
AI4PRIVACY_LABEL_MAP = {
    "FIRSTNAME": "private_person", "LASTNAME": "private_person",
    "MIDDLENAME": "private_person", "FULLNAME": "private_person",
    "TITLE": "private_person", "SUFFIX": "private_person",
    "ALIAS": "private_person", "USERNAME": "private_person",
    "GENDER": "private_person",
    "STREET": "private_address", "CITY": "private_address",
    "STATE": "private_address", "ZIPCODE": "private_address",
    "COUNTRY": "private_address", "COUNTY": "private_address",
    "POBOX": "private_address", "FULLADDRESS": "private_address",
    "BUILDINGNUMBER": "private_address", "SECONDARYADDRESS": "private_address",
    "EMAIL": "private_email",
    "PHONENUMBER": "private_phone", "PHONE": "private_phone",
    "URL": "private_url", "IP": "private_url", "IPADDRESS": "private_url",
    "DATE": "private_date", "DOB": "private_date", "TIME": "private_date",
    "CREDITCARDNUMBER": "account_number", "ACCOUNTNUMBER": "account_number",
    "IBAN": "account_number", "BIC": "account_number",
    "BITCOINADDRESS": "account_number",
    "SSN": "secret", "PASSPORT": "secret",
    "DRIVERLICENSE": "secret", "PIN": "secret", "PASSWORD": "secret",
}


def run_model_on_texts(pipe, texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = pipe(batch)
        results.extend(preds)
    return results



def _span_f1(gold_spans, pred_spans):
    """
    Compute span-level precision, recall, F1 per entity type.
    gold_spans / pred_spans: list of (start, end, entity_type).
    A predicted span is a TP if it matches a gold span on (start, end, type).
    """
    from collections import defaultdict
    gold_by_type = defaultdict(set)
    pred_by_type = defaultdict(set)
    for s, e, t in gold_spans:
        gold_by_type[t].add((s, e))
    for s, e, t in pred_spans:
        pred_by_type[t].add((s, e))

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
        report[t] = {"precision": prec, "recall": rec, "f1-score": f1,
                     "support": len(g)}
        total_tp += tp; total_fp += fp; total_fn += fn

    # micro average
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) else 0.0)
    report["micro avg"] = {"precision": micro_prec, "recall": micro_rec,
                           "f1-score": micro_f1,
                           "support": sum(len(v) for v in gold_by_type.values())}
    return report


def evaluate_on_ai4privacy(pipe, split="validation", max_samples=None):
    """
    AI4Privacy v2 format: source_text + privacy_mask (char-offset spans).
    Each privacy_mask entry: {"value": str, "label": str, "start": int, "stop": int}
    """
    print(f"\nLoading AI4Privacy ({split} split)...")
    ds = load_dataset("ai4privacy/pii-masking-400k", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts = [ex["source_text"] for ex in ds]
    print(f"Running model on {len(texts)} examples...")
    raw_preds = run_model_on_texts(pipe, texts)

    all_gold, all_pred = [], []
    for ex, pred_ents in zip(ds, raw_preds):
        # Gold spans from privacy_mask
        gold = []
        for m in ex["privacy_mask"]:
            mapped = AI4PRIVACY_LABEL_MAP.get(m["label"])
            if mapped:
                end = m.get("stop", m.get("end"))
                gold.append((m["start"], end, mapped))

        # Predicted spans from model
        pred = [(e["start"], e["end"], e["entity_group"]) for e in pred_ents]

        all_gold.append(gold)
        all_pred.append(pred)

    # Aggregate across all examples
    flat_gold = [s for spans in all_gold for s in spans]
    flat_pred = [s for spans in all_pred for s in spans]
    return _span_f1(flat_gold, flat_pred)


def evaluate_on_mimic(pipe, mimic_dir, max_samples=500):
    """No gold labels — reports entity detection statistics only."""
    mimic_path = Path(mimic_dir)
    note_file = mimic_path / "note" / "discharge.csv.gz"
    if not note_file.exists():
        print(f"MIMIC note file not found at {note_file}. Skipping MIMIC eval.")
        return None

    import pandas as pd
    print(f"\nLoading MIMIC discharge notes (sample={max_samples})...")
    df = pd.read_csv(note_file, compression="gzip", nrows=max_samples, usecols=["text"])
    texts = df["text"].dropna().tolist()
    texts = [" ".join(t.split()[:512]) for t in texts]  # truncate for CPU speed

    print(f"Running model on {len(texts)} MIMIC notes...")
    raw_preds = run_model_on_texts(pipe, texts, batch_size=8)

    counts, total = {}, 0
    for pred_ents in raw_preds:
        for ent in pred_ents:
            etype = ent["entity_group"]
            counts[etype] = counts.get(etype, 0) + 1
            total += 1

    return {
        "total_notes": len(texts),
        "total_entities_detected": total,
        "entities_per_note": total / len(texts),
        "entity_type_counts": counts,
    }


def print_f1_table(report, domain):
    print(f"\n{'='*55}")
    print(f"  F1 Results — {domain}")
    print(f"{'='*55}")
    print(f"  {'Entity':<22} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print(f"  {'-'*46}")
    for label, vals in report.items():
        if label in ("micro avg", "macro avg", "weighted avg") or not isinstance(vals, dict):
            continue
        print(f"  {label:<22} {vals['precision']:>6.3f} {vals['recall']:>6.3f} "
              f"{vals['f1-score']:>6.3f} {int(vals['support']):>6}")
    print(f"  {'-'*46}")
    for avg in ("micro avg", "macro avg"):
        if avg in report:
            v = report[avg]
            print(f"  {avg:<22} {v['precision']:>6.3f} {v['recall']:>6.3f} "
                  f"{v['f1-score']:>6.3f} {int(v['support']):>6}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-dir", default="data/mimic-iv-note")
    parser.add_argument("--max-ai4privacy", type=int, default=None,
                        help="Cap AI4Privacy samples (e.g. 500 for a quick run)")
    parser.add_argument("--max-mimic", type=int, default=500)
    parser.add_argument("--output", default="results/baseline_eval.json")
    parser.add_argument("--device", type=int, default=-1,
                        help="Device index: -1 for CPU, 0 for first GPU")
    args = parser.parse_args()

    print("Loading openai/privacy-filter...")
    pipe = pipeline(
        "token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
        device=args.device,
    )

    results = {}

    ai4p_report = evaluate_on_ai4privacy(pipe, max_samples=args.max_ai4privacy)
    print_f1_table(ai4p_report, "General domain (AI4Privacy)")
    results["ai4privacy"] = ai4p_report

    mimic_stats = evaluate_on_mimic(pipe, args.mimic_dir, max_samples=args.max_mimic)
    if mimic_stats:
        print(f"\n{'='*55}")
        print(f"  Entity Detection Stats — Clinical domain (MIMIC)")
        print(f"{'='*55}")
        print(f"  Notes evaluated:      {mimic_stats['total_notes']}")
        print(f"  Total entities found: {mimic_stats['total_entities_detected']}")
        print(f"  Entities per note:    {mimic_stats['entities_per_note']:.2f}")
        print(f"  By type: {mimic_stats['entity_type_counts']}")
        results["mimic"] = mimic_stats

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
