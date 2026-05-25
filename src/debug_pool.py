"""
debug_pool.py

Quick diagnostic: print all label types present in the AI4Privacy privacy_mask field,
and test the pattern matching on a few MIMIC notes.

Run:
    python src/debug_pool.py
"""

import csv
import re
from datasets import load_dataset
from build_clinical_eval import PLACEHOLDER_PATTERNS, ENTITY_POOL_KEYS, build_entity_pool, inject_pii
import random

# ── 1. Check actual label names in AI4Privacy ────────────────────────────────
print("=== AI4Privacy label types (first 500 validation examples) ===")
ds = load_dataset("ai4privacy/pii-masking-400k", split="validation")
ds = ds.select(range(500))

label_counts: dict[str, int] = {}
for ex in ds:
    for m in ex["privacy_mask"]:
        label = m["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

for label, count in sorted(label_counts.items()):
    in_pool = any(label in keys for keys in ENTITY_POOL_KEYS.values())
    flag = "✓" if in_pool else "✗ NOT IN POOL"
    print(f"  {label:<30} {count:>5}  {flag}")

# ── 2. Test pattern matching on first MIMIC note ─────────────────────────────
print("\n=== Pattern matching on first MIMIC note ===")
MIMIC_CSV = ("/mmfs1/gscratch/scrubbed/jieyao24/data/mimic-iv-note/"
             "physionet.org/files/mimic-iv-note/2.2/note/discharge.csv")
with open(MIMIC_CSV) as f:
    reader = csv.DictReader(f)
    row = next(reader)
    text = row["text"]

print("First 400 chars of note:")
print(text[:400])
print()

for pattern, entity_type in PLACEHOLDER_PATTERNS:
    matches = list(re.finditer(pattern, text))
    if matches:
        for m in matches[:2]:
            print(f"  MATCHED [{entity_type}] pattern={pattern!r}")
            print(f"    full match: {m.group()!r}")
            print(f"    group(1):  {m.group(1)!r}")
            print(f"    group(2):  {m.group(2)!r}")
    else:
        # Check if ___ appears near expected context
        kw = pattern.split("(")[1].split("\\")[0].replace("(?i)","").strip()
        print(f"  NO MATCH  [{entity_type}] pattern={pattern!r}")

# ── 3. Test sample_entity for each type ──────────────────────────────────────
print("\n=== sample_entity availability ===")
pool = build_entity_pool(max_examples=500)
rng = random.Random(42)

from build_clinical_eval import sample_entity
for etype, keys in ENTITY_POOL_KEYS.items():
    val = sample_entity(pool, etype, rng)
    pool_sizes = {k: len(pool.get(k, [])) for k in keys}
    print(f"  {etype:<20} keys={pool_sizes}  sample={val!r}")
