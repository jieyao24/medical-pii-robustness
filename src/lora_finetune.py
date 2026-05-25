"""
lora_finetune.py

LoRA fine-tuning of openai/privacy-filter on MIMIC-IV-Note discharge summaries.
Adapts attention projections only (q_proj, v_proj) — safer for sparse MoE models
than adapting expert layers, which risks router instability.

Ablation axes:
  --lora-rank   : 4 | 8 | 16
  --train-size  : 1000 | 5000 | 20000

Data split (from data/mimic_pii_injected.jsonl, 22k total):
  Train : first --train-size notes
  Val   : notes 20001–21000  (1k held-out, never used for training)
  Test  : notes 21001–22000  (1k held-out, final eval only)

Catastrophic forgetting is monitored at every checkpoint by evaluating on a
held-out AI4Privacy split (2k examples).

Usage:
    python src/lora_finetune.py \
        --lora-rank 8 \
        --train-size 5000 \
        --output-dir results/checkpoints/lora_r8_n5000 \
        --device 0
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    pipeline,
)

from baseline_eval import evaluate_on_ai4privacy, _span_f1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "openai/privacy-filter"
INJECTED_JSONL = "data/mimic_pii_injected.jsonl"

# Indices for the fixed val/test split (always held-out regardless of train size)
VAL_START  = 20000
VAL_END    = 21000
TEST_START = 21000
TEST_END   = 22000

MAX_SEQ_LEN = 512  # tokens; matches GPU memory budget on RTX 6000


# ---------------------------------------------------------------------------
# Data loading and tokenization
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def align_labels_to_tokens(text: str, entities: list[dict],
                            tokenizer, label2id: dict) -> dict:
    """
    Tokenize text and produce BIOES token labels aligned to subword tokens.
    Returns a dict with input_ids, attention_mask, labels (all lists).
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    n_tokens = len(enc["input_ids"])

    # Default: O label
    o_id = label2id.get("O", 0)
    labels = [o_id] * n_tokens

    # Sort entities left-to-right
    for ent in sorted(entities, key=lambda e: e["start"]):
        etype = ent["entity_type"]
        char_start, char_end = ent["start"], ent["end"]

        # Find token indices that overlap this char span
        tok_indices = [
            i for i, (ts, te) in enumerate(offsets)
            if ts is not None and te is not None and ts < char_end and te > char_start
        ]
        if not tok_indices:
            continue

        if len(tok_indices) == 1:
            label_str = f"S-{etype}"
        else:
            label_str = None  # use B/I/E below

        for j, tok_i in enumerate(tok_indices):
            if label_str:                        # single-token span
                lid = label2id.get(label_str, o_id)
            elif j == 0:
                lid = label2id.get(f"B-{etype}", o_id)
            elif j == len(tok_indices) - 1:
                lid = label2id.get(f"E-{etype}", o_id)
            else:
                lid = label2id.get(f"I-{etype}", o_id)
            labels[tok_i] = lid

    # Mask special tokens (offset == (0,0) for [CLS]/[SEP]/padding)
    for i, (ts, te) in enumerate(offsets):
        if ts == 0 and te == 0:
            labels[i] = -100  # ignored in cross-entropy

    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


def build_hf_dataset(records: list[dict], tokenizer, label2id: dict) -> Dataset:
    tokenized = [
        align_labels_to_tokens(r["text"], r["entities"], tokenizer, label2id)
        for r in records
    ]
    return Dataset.from_list(tokenized)


# ---------------------------------------------------------------------------
# Catastrophic forgetting callback
# ---------------------------------------------------------------------------

class ForgettingCallback(TrainerCallback):
    """Evaluate on held-out AI4Privacy at the end of each epoch."""

    def __init__(self, tokenizer, device: int, output_dir: str, max_samples: int = 2000):
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.history = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n[ForgettingCallback] Evaluating on AI4Privacy after epoch {state.epoch:.0f}...")
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
        )
        report = evaluate_on_ai4privacy(pipe, max_samples=self.max_samples)
        micro = report.get("micro avg", {})
        f1 = micro.get("f1-score", 0.0)
        print(f"  AI4Privacy micro F1 after epoch {state.epoch:.0f}: {f1:.4f}")
        self.history.append({"epoch": state.epoch, "ai4privacy_micro_f1": f1})

        # Save history
        hist_path = Path(self.output_dir) / "forgetting_history.json"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def build_lora_model(lora_rank: int):
    """Load the base model and wrap with LoRA on attention projections only."""
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

    lora_cfg = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_rank,
        lora_alpha=lora_rank * 2,   # alpha = 2r is a common default
        lora_dropout=0.1,
        bias="none",
        # Attention projections only — avoids MoE expert layer instability
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-rank",   type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--train-size",  type=int, default=5000,
                        help="Number of MIMIC notes to use for training (1000/5000/20000)")
    parser.add_argument("--injected-data", default=INJECTED_JSONL)
    parser.add_argument("--output-dir",  default=None,
                        help="Where to save checkpoints (default: auto-named)")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--device",      type=int, default=0)
    parser.add_argument("--forgetting-samples", type=int, default=2000,
                        help="AI4Privacy examples for catastrophic forgetting eval")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/checkpoints/lora_r{args.lora_rank}_n{args.train_size}"

    print(f"Config: LoRA rank={args.lora_rank}, train_size={args.train_size}, "
          f"epochs={args.epochs}, lr={args.lr}")
    print(f"Output: {args.output_dir}")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.injected_data}...")
    all_records = load_jsonl(args.injected_data)
    print(f"Total records: {len(all_records)}")

    if len(all_records) < TEST_END:
        raise ValueError(
            f"Need at least {TEST_END} records for fixed val/test split, "
            f"got {len(all_records)}. Re-run build_clinical_eval.py with --max-notes 22000."
        )
    if args.train_size > VAL_START:
        raise ValueError(
            f"--train-size {args.train_size} overlaps the fixed val/test split "
            f"(starts at {VAL_START}). Use at most {VAL_START}."
        )

    train_records = all_records[:args.train_size]
    val_records   = all_records[VAL_START:VAL_END]
    test_records  = all_records[TEST_START:TEST_END]
    print(f"Split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    # ── Tokenizer and label map ──────────────────────────────────────────────
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load label2id from the base model config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID)
    label2id = config.label2id
    id2label = config.id2label
    print(f"Label set: {len(label2id)} labels")

    # ── Build HuggingFace datasets ───────────────────────────────────────────
    print("\nTokenizing train split...")
    train_ds = build_hf_dataset(train_records, tokenizer, label2id)
    print("Tokenizing val split...")
    val_ds   = build_hf_dataset(val_records, tokenizer, label2id)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_lora_model(args.lora_rank)
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    model = model.to(device)

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        # A40/A100/H200 support bf16 natively; RTX 6000/2080Ti need fp16.
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        report_to="none",
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    forgetting_cb = ForgettingCallback(
        tokenizer=tokenizer,
        device=args.device,
        output_dir=args.output_dir,
        max_samples=args.forgetting_samples,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[forgetting_cb],
    )

    print("\nStarting training...")
    trainer.train()

    # ── Final test evaluation ────────────────────────────────────────────────
    print("\nRunning final test evaluation on clinical (PII-injected MIMIC)...")
    test_ds = build_hf_dataset(test_records, tokenizer, label2id)
    test_results = trainer.evaluate(test_ds)
    print(f"Test results: {test_results}")

    # Save final results summary
    summary = {
        "lora_rank":   args.lora_rank,
        "train_size":  args.train_size,
        "epochs":      args.epochs,
        "lr":          args.lr,
        "test_eval":   test_results,
        "forgetting":  forgetting_cb.history,
    }
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Save LoRA adapter weights
    adapter_path = Path(args.output_dir) / "adapter"
    model.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
