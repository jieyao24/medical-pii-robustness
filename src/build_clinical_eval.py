"""
build_clinical_eval.py

Build a labeled clinical evaluation set by injecting known PII entities from
AI4Privacy into de-identified MIMIC-IV-Note discharge summaries.

MIMIC-IV de-identification replaces PII with ___ (three or more underscores).
We detect ___ placeholders using surrounding context to determine PII type,
then replace them with real entities drawn from the AI4Privacy validation split.

Output: data/mimic_pii_injected.jsonl  (gitignored, Hyak local only)
Each line: {"note_id": ..., "text": ..., "entities": [{"start": int, "end": int,
            "entity_type": str, "value": str}, ...]}

Usage:
    python src/build_clinical_eval.py \
        --mimic-csv /path/to/discharge.csv.gz \
        --max-notes 5000 \
        --output data/mimic_pii_injected.jsonl
"""

import argparse
import csv
import gzip
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Placeholder detection patterns
# Each entry: (regex_pattern, entity_type, capture_group_for_replacement)
# Patterns are ordered most-specific first.
# ---------------------------------------------------------------------------
PLACEHOLDER_PATTERNS = [
    # Patient header fields
    (r"(?i)(Name\s*:\s*)(___+)",                        "private_person"),
    (r"(?i)(Attending\s*:\s*)(___+)",                   "private_person"),
    (r"(?i)((?:Mrs?|Ms|Dr|Prof)\.\s+)(___+)",           "private_person"),
    (r"(?i)(Dear\s+(?:Ms?\.|Mrs?\.|Dr\.\s+)?)(___+)",   "private_person"),
    (r"(?i)(Unit\s*No\s*:\s*)(___+)",                   "account_number"),
    (r"(?i)(Admission\s*Date\s*:\s*)(___+)",            "private_date"),
    (r"(?i)(Discharge\s*Date\s*:\s*)(___+)",            "private_date"),
    (r"(?i)(Date\s*of\s*Birth\s*:\s*)(___+)",           "private_date"),
    (r"(?i)(Service\s*:\s*)(___+)",                     None),              # skip — not PII
    # In-sentence person references
    (r"(?i)((?:followed by |scheduled with |with |by )Dr\.\s+)(___+)",    "private_person"),
    (r"(?i)((?:Mrs?|Ms)\.\s+)(___+)",                   "private_person"),
]

# entity_type → key in the AI4Privacy pool dict
# NOTE: AI4Privacy uses GIVENNAME/SURNAME (not FIRSTNAME/LASTNAME),
#       DATEOFBIRTH (not DOB/DATE), ACCOUNTNUM (not ACCOUNTNUMBER),
#       TELEPHONENUM (not PHONENUMBER), SOCIALNUM (not SSN).
ENTITY_POOL_KEYS = {
    "private_person":  ["GIVENNAME", "SURNAME"],
    "private_date":    ["DATEOFBIRTH"],
    "account_number":  ["ACCOUNTNUM", "CREDITCARDNUMBER"],
    "private_address": ["STREET", "CITY", "BUILDINGNUM", "ZIPCODE"],
    "private_email":   ["EMAIL"],
    "private_phone":   ["TELEPHONENUM"],
    "secret":          ["SOCIALNUM", "DRIVERLICENSENUM", "PASSWORD", "TAXNUM", "IDCARDNUM"],
}


# ---------------------------------------------------------------------------
# Build entity pool from AI4Privacy
# ---------------------------------------------------------------------------

def build_entity_pool(max_examples: int = 5000) -> dict[str, list[str]]:
    """Extract a pool of real PII values from AI4Privacy by label type."""
    print(f"Loading AI4Privacy to build entity pool (max={max_examples})...")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="validation")
    ds = ds.select(range(min(max_examples, len(ds))))

    pool: dict[str, list[str]] = {}
    for ex in ds:
        text = ex["source_text"]
        for m in ex["privacy_mask"]:
            label = m["label"]
            start, end = m["start"], m.get("stop", m.get("end"))
            value = text[start:end].strip()
            if value:
                pool.setdefault(label, []).append(value)

    total = sum(len(v) for v in pool.values())
    print(f"Entity pool: {len(pool)} label types, {total} total values")
    return pool


def sample_entity(pool: dict, entity_type: str, rng: random.Random) -> str | None:
    """Sample a random entity value for the given model entity type."""
    keys = ENTITY_POOL_KEYS.get(entity_type, [])
    candidates = []
    for k in keys:
        candidates.extend(pool.get(k, []))
    if not candidates:
        return None
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Injection logic
# ---------------------------------------------------------------------------

def inject_pii(note_id: str, text: str, pool: dict, rng: random.Random) -> dict | None:
    """
    Find contextual ___ placeholders in the note and replace them with real PII.
    Returns a dict with text (modified) and entity list, or None if no injections made.

    We process matches right-to-left so earlier offsets stay valid after each replacement.
    """
    # Collect all candidate replacements: (start, end, entity_type, replacement_value)
    candidates = []
    seen_spans = set()  # avoid double-matching the same ___ span

    for pattern, entity_type in PLACEHOLDER_PATTERNS:
        if entity_type is None:
            continue
        for m in re.finditer(pattern, text):
            # Group 2 is always the ___ span
            placeholder_start = m.start(2)
            placeholder_end = m.end(2)
            span_key = (placeholder_start, placeholder_end)
            if span_key in seen_spans:
                continue
            value = sample_entity(pool, entity_type, rng)
            if value is None:
                continue
            seen_spans.add(span_key)
            candidates.append((placeholder_start, placeholder_end, entity_type, value))

    if not candidates:
        return None

    # Sort right-to-left so replacements don't shift subsequent offsets
    candidates.sort(key=lambda x: x[0], reverse=True)

    entities = []
    modified_text = text

    for p_start, p_end, entity_type, value in candidates:
        modified_text = modified_text[:p_start] + value + modified_text[p_end:]
        # Offset of this entity in the final text will be adjusted below
        entities.append({
            "orig_start": p_start,
            "orig_end":   p_end,
            "entity_type": entity_type,
            "value":       value,
        })

    # Recompute final char offsets (right-to-left replacements mean we need a forward pass)
    # We stored original offsets and process in reverse, so rebuild forward.
    # Simpler: just re-find each injected value at its expected position.
    # Since we replaced right-to-left, positions to the left are still valid.
    # Reconstruct forward:
    entities.sort(key=lambda x: x["orig_start"])  # back to left-to-right order

    # Walk through text to compute final offsets
    offset_delta = 0
    final_entities = []
    for e in entities:
        orig_len = e["orig_end"] - e["orig_start"]
        new_len = len(e["value"])
        final_start = e["orig_start"] + offset_delta
        final_end = final_start + new_len
        final_entities.append({
            "start":       final_start,
            "end":         final_end,
            "entity_type": e["entity_type"],
            "value":       e["value"],
        })
        offset_delta += (new_len - orig_len)

    return {
        "note_id": note_id,
        "text": modified_text,
        "entities": final_entities,
    }


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_offsets(record: dict) -> bool:
    """Sanity-check that every entity's value matches text[start:end]."""
    text = record["text"]
    for e in record["entities"]:
        extracted = text[e["start"]:e["end"]]
        if extracted != e["value"]:
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-csv",
                        default="/mmfs1/gscratch/scrubbed/jieyao24/data/mimic-iv-note/"
                                "physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz")
    parser.add_argument("--max-notes", type=int, default=5000,
                        help="Max MIMIC notes to process")
    parser.add_argument("--pool-size", type=int, default=5000,
                        help="AI4Privacy examples to build entity pool from")
    parser.add_argument("--output", default="data/mimic_pii_injected.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Build entity pool
    pool = build_entity_pool(max_examples=args.pool_size)

    # Open MIMIC CSV (supports both .gz and plain)
    mimic_path = Path(args.mimic_csv)
    opener = gzip.open if mimic_path.suffix == ".gz" else open
    open_mode = "rt"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_injected = 0
    n_verify_fail = 0

    print(f"Processing MIMIC notes (max={args.max_notes})...")
    with opener(mimic_path, open_mode) as f, open(args.output, "w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            if n_processed >= args.max_notes:
                break
            n_processed += 1

            note_id = row.get("note_id", str(n_processed))
            text = row.get("text", "")
            if not text.strip():
                continue

            result = inject_pii(note_id, text, pool, rng)
            if result is None:
                continue  # no contextual placeholders found

            if not verify_offsets(result):
                n_verify_fail += 1
                continue  # skip records with offset bugs

            out.write(json.dumps(result) + "\n")
            n_injected += 1

            if n_injected % 500 == 0:
                print(f"  {n_processed} notes processed, {n_injected} injected...")

    print(f"\nDone.")
    print(f"  Notes processed:       {n_processed}")
    print(f"  Notes with injections: {n_injected}")
    print(f"  Offset verify failures:{n_verify_fail}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
