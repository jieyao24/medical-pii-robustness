"""
Evasion attack suite for openai/privacy-filter.

Attack taxonomy:
  Character-level — operate below tokenizer, expected to survive fine-tuning:
    homoglyph  : Replace Latin chars with Cyrillic/Greek look-alikes
    zwsp       : Insert zero-width spaces between chars inside PII spans
    whitespace : Insert thin/hair spaces inside PII spans

  Domain-level — exploit clinical distribution shift, expected to be fixed by fine-tuning:
    abbreviation : Shorten person names (John Williams -> J. Williams, Williams J.)
    honorific    : Prepend clinical titles (Dr., Pt., Attending:)
    field_format : Embed names in structured clinical field notation (Attending: <name>)

Evasion is measured as:
  entity was correctly detected on clean text, then missed after perturbation.
  Evasion rate = evaded / originally_detected  (per attack × entity type)
"""

import argparse
import json
import random
import unicodedata
from pathlib import Path

from datasets import load_dataset
from transformers import pipeline


# ---------------------------------------------------------------------------
# Homoglyph table: Latin → visually identical Unicode (Cyrillic/Greek/etc.)
# ---------------------------------------------------------------------------
HOMOGLYPHS = {
    "a": "а",  # Cyrillic а
    "e": "е",  # Cyrillic е
    "o": "о",  # Cyrillic о
    "p": "р",  # Cyrillic р
    "c": "с",  # Cyrillic с
    "x": "х",  # Cyrillic х
    "i": "і",  # Ukrainian і
    "A": "А",  # Cyrillic А
    "B": "В",  # Cyrillic В
    "C": "С",  # Cyrillic С
    "E": "Е",  # Cyrillic Е
    "H": "Н",  # Cyrillic Н
    "K": "К",  # Cyrillic К
    "M": "М",  # Cyrillic М
    "O": "О",  # Cyrillic О
    "P": "Р",  # Cyrillic Р
    "T": "Т",  # Cyrillic Т
    "X": "Х",  # Cyrillic Х
}

ZWSP = "​"       # zero-width space
THIN_SPACE = " " # thin space

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


# ---------------------------------------------------------------------------
# Perturbation functions  (each takes a span string, returns perturbed string)
# ---------------------------------------------------------------------------

def attack_homoglyph(span: str) -> str:
    """Replace substitutable Latin chars with Cyrillic look-alikes."""
    out = []
    replaced = False
    for ch in span:
        if ch in HOMOGLYPHS:
            out.append(HOMOGLYPHS[ch])
            replaced = True
        else:
            out.append(ch)
    return "".join(out) if replaced else span + ZWSP  # fallback if no substitutable chars


def attack_zwsp(span: str) -> str:
    """Insert zero-width spaces between every character."""
    return ZWSP.join(list(span))


def attack_whitespace(span: str) -> str:
    """Insert thin spaces between every character."""
    return THIN_SPACE.join(list(span))


def attack_abbreviation(span: str) -> str:
    """
    Abbreviate person names: 'John Williams' -> 'J. Williams'.
    Falls back to inserting a period after the first letter if no space found.
    """
    parts = span.strip().split()
    if len(parts) >= 2:
        abbrev = ". ".join(p[0] for p in parts[:-1])
        return f"{abbrev}. {parts[-1]}"
    elif len(parts) == 1 and len(parts[0]) > 1:
        return parts[0][0] + ". " + parts[0][1:]
    return span


def attack_honorific(span: str) -> str:
    """Prepend a clinical honorific."""
    honorifics = ["Dr. ", "Pt. ", "Attn: ", "RN "]
    return random.choice(honorifics) + span


def attack_field_format(span: str) -> str:
    """Embed in a structured clinical field."""
    fields = ["Attending: ", "Patient: ", "Provider: ", "Referring MD: "]
    return random.choice(fields) + span


ATTACKS = {
    # character-level
    "homoglyph":    (attack_homoglyph,   {"private_person", "private_address", "private_email"}),
    "zwsp":         (attack_zwsp,        {"private_person", "private_address", "private_email",
                                          "private_phone", "account_number", "secret"}),
    "whitespace":   (attack_whitespace,  {"private_person", "private_address"}),
    # domain-level
    "abbreviation": (attack_abbreviation, {"private_person"}),
    "honorific":    (attack_honorific,    {"private_person"}),
    "field_format": (attack_field_format, {"private_person", "private_address"}),
}


# ---------------------------------------------------------------------------
# Span detection helpers
# ---------------------------------------------------------------------------

def get_detected_spans(pipe, text: str) -> list[dict]:
    """Return entity dicts from the pipeline with start/end char offsets."""
    return pipe(text)


def spans_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def entity_detected(pred_spans, gold_start: int, gold_end: int, gold_type: str) -> bool:
    """True if any predicted span overlaps the gold span with the right type."""
    for p in pred_spans:
        if p["entity_group"] == gold_type and spans_overlap(
            p["start"], p["end"], gold_start, gold_end
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_attack(pipe, examples: list[dict], attack_name: str,
                    attack_fn, target_types: set) -> dict:
    """
    For each example with a gold entity of the target type:
      1. Check baseline detection on clean text
      2. Perturb the entity span in the text
      3. Check detection on perturbed text
    Returns per-entity-type evasion counts.
    """
    counts = {}  # entity_type -> {originally_detected, evaded}

    for ex in examples:
        text = ex["text"]
        for gold in ex["entities"]:
            etype = gold["entity_type"]
            if etype not in target_types:
                continue
            if etype not in counts:
                counts[etype] = {"originally_detected": 0, "evaded": 0, "total": 0}

            gstart, gend = gold["start"], gold["end"]
            span = text[gstart:gend]

            # Baseline: was the entity detected on clean text?
            clean_preds = get_detected_spans(pipe, text)
            detected_clean = entity_detected(clean_preds, gstart, gend, etype)
            counts[etype]["total"] += 1
            if not detected_clean:
                continue  # can't measure evasion on already-missed entities

            counts[etype]["originally_detected"] += 1

            # Perturb the span in the text
            perturbed_span = attack_fn(span)
            perturbed_text = text[:gstart] + perturbed_span + text[gend:]

            # New offsets after perturbation (span may have grown)
            new_end = gstart + len(perturbed_span)
            pert_preds = get_detected_spans(pipe, perturbed_text)
            detected_pert = entity_detected(pert_preds, gstart, new_end, etype)

            if not detected_pert:
                counts[etype]["evaded"] += 1

    # Compute evasion rates
    for etype, c in counts.items():
        od = c["originally_detected"]
        c["evasion_rate"] = round(c["evaded"] / od, 4) if od > 0 else None

    return counts


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_ai4privacy_examples(max_samples: int | None = None) -> list[dict]:
    """Load AI4Privacy and convert to flat entity list format."""
    print(f"Loading AI4Privacy (max={max_samples})...")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="validation")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for ex in ds:
        text = ex["source_text"]
        entities = []
        for m in ex["privacy_mask"]:
            mapped = AI4PRIVACY_LABEL_MAP.get(m["label"])
            if mapped:
                entities.append({"entity_type": mapped, "start": m["start"], "end": m.get("stop", m.get("end"))})
        if entities:
            examples.append({"text": text, "entities": entities})

    return examples


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_evasion_table(results: dict):
    print(f"\n{'='*65}")
    print(f"  {'Attack':<16} {'Entity Type':<22} {'Det':>5} {'Evd':>5} {'Rate':>7}")
    print(f"  {'-'*59}")
    for attack_name, per_type in results.items():
        for etype, c in per_type.items():
            rate = f"{c['evasion_rate']:.1%}" if c["evasion_rate"] is not None else "  N/A"
            print(f"  {attack_name:<16} {etype:<22} {c['originally_detected']:>5} "
                  f"{c['evaded']:>5} {rate:>7}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Number of AI4Privacy examples to evaluate")
    parser.add_argument("--attacks", nargs="+", default=list(ATTACKS.keys()),
                        choices=list(ATTACKS.keys()),
                        help="Which attacks to run (default: all)")
    parser.add_argument("--model", default="openai/privacy-filter")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter dir. If omitted, uses base model.")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--output", default="results/attack_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.adapter:
        print(f"Loading base model {args.model} + LoRA adapter from {args.adapter}...")
        from peft import PeftModel
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        base = AutoModelForTokenClassification.from_pretrained(args.model)
        model = PeftModel.from_pretrained(base, args.adapter)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=args.device,
        )
        run_label = f"LoRA ({args.adapter})"
    else:
        print(f"Loading base model {args.model}...")
        pipe = pipeline(
            "token-classification",
            model=args.model,
            aggregation_strategy="simple",
            device=args.device,
        )
        run_label = f"Baseline ({args.model})"

    print(f"Run: {run_label}")

    examples = load_ai4privacy_examples(max_samples=args.max_samples)
    print(f"Loaded {len(examples)} examples with PII entities.\n")

    results = {}
    for attack_name in args.attacks:
        attack_fn, target_types = ATTACKS[attack_name]
        print(f"Running attack: {attack_name}  (targets: {sorted(target_types)})")
        results[attack_name] = evaluate_attack(pipe, examples, attack_name, attack_fn, target_types)

    print_evasion_table(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = {"run_label": run_label, "attacks": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
