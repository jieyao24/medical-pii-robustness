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
# NOTE: AI4Privacy v2 (pii-masking-400k) uses GIVENNAME/SURNAME (not FIRSTNAME/LASTNAME),
# DATEOFBIRTH (not DOB/DATE), ACCOUNTNUM (not ACCOUNTNUMBER), TELEPHONENUM (not PHONENUMBER),
# SOCIALNUM/DRIVERLICENSENUM (not SSN/DRIVERLICENSE). Both old and actual labels are
# included below so this map works regardless of dataset version.
AI4PRIVACY_LABEL_MAP = {
    # Person — actual v2 labels first, legacy fallbacks retained
    "GIVENNAME": "private_person", "SURNAME": "private_person",
    "FIRSTNAME": "private_person", "LASTNAME": "private_person",
    "MIDDLENAME": "private_person", "FULLNAME": "private_person",
    "TITLE": "private_person", "SUFFIX": "private_person",
    "ALIAS": "private_person", "USERNAME": "private_person",
    "GENDER": "private_person",
    # Address
    "STREET": "private_address", "CITY": "private_address",
    "STATE": "private_address", "ZIPCODE": "private_address",
    "COUNTRY": "private_address", "COUNTY": "private_address",
    "POBOX": "private_address", "FULLADDRESS": "private_address",
    "BUILDINGNUM": "private_address", "BUILDINGNUMBER": "private_address",
    "SECONDARYADDRESS": "private_address",
    # Email
    "EMAIL": "private_email",
    # Phone — actual v2 label + legacy
    "TELEPHONENUM": "private_phone",
    "PHONENUMBER": "private_phone", "PHONE": "private_phone",
    # URL
    "URL": "private_url", "IP": "private_url", "IPADDRESS": "private_url",
    # Date — actual v2 label + legacy
    "DATEOFBIRTH": "private_date",
    "DATE": "private_date", "DOB": "private_date", "TIME": "private_date",
    # Account number — actual v2 label + legacy
    "ACCOUNTNUM": "account_number",
    "CREDITCARDNUMBER": "account_number", "ACCOUNTNUMBER": "account_number",
    "IBAN": "account_number", "BIC": "account_number",
    "BITCOINADDRESS": "account_number",
    # Secret — actual v2 labels + legacy
    "SOCIALNUM": "secret", "DRIVERLICENSENUM": "secret",
    "TAXNUM": "secret", "IDCARDNUM": "secret",
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



def _span_f1(all_gold, all_pred):
    """
    Compute span-level F1 with OVERLAP matching per entity type.

    all_gold / all_pred: list-of-lists, one inner list per document.
    Each span: (start, end, entity_type).

    A predicted span is a TP if it overlaps a gold span of the same type
    in the same document (greedy left-to-right; each gold matched at most once).

    Overlap matching is used (not exact char-offset matching) because subword
    tokenizer pipelines frequently produce offsets that are ±1 char from the
    gold annotation boundary. Exact matching would give near-zero F1 despite
    correct detection, while overlap matching faithfully measures coverage.
    """
    from collections import defaultdict

    # Group by (doc_i, type) → list of (start, end)
    gold_by_type_doc = defaultdict(lambda: defaultdict(list))
    pred_by_type_doc = defaultdict(lambda: defaultdict(list))
    gold_type_counts = defaultdict(int)
    pred_type_counts = defaultdict(int)

    for doc_i, spans in enumerate(all_gold):
        for s, e, t in spans:
            gold_by_type_doc[t][doc_i].append((s, e))
            gold_type_counts[t] += 1

    for doc_i, spans in enumerate(all_pred):
        for s, e, t in spans:
            pred_by_type_doc[t][doc_i].append((s, e))
            pred_type_counts[t] += 1

    all_types = set(gold_type_counts) | set(pred_type_counts)
    report = {}
    total_tp = total_fp = total_fn = 0

    for t in all_types:
        g_docs = gold_by_type_doc[t]
        p_docs = pred_by_type_doc[t]
        n_gold = gold_type_counts[t]
        n_pred = pred_type_counts[t]
        tp = 0

        for doc_i in set(g_docs) | set(p_docs):
            golds = g_docs[doc_i]
            preds = p_docs[doc_i]
            matched = set()
            for ps, pe in preds:
                for gi, (gs, ge) in enumerate(golds):
                    if gi not in matched and ps < ge and gs < pe:  # overlap
                        matched.add(gi)
                        break
            tp += len(matched)

        fp = n_pred - tp
        fn = n_gold - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report[t] = {"precision": prec, "recall": rec, "f1-score": f1,
                     "support": n_gold}
        total_tp += tp; total_fp += fp; total_fn += fn

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) else 0.0)
    report["micro avg"] = {"precision": micro_prec, "recall": micro_rec,
                           "f1-score": micro_f1,
                           "support": sum(gold_type_counts.values())}
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

    # Pass list-of-lists directly; _span_f1 handles per-document grouping.
    return _span_f1(all_gold, all_pred)


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
