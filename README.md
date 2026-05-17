# Medical PII Robustness

Adversarial robustness evaluation of PII detection under clinical domain shift, with LoRA fine-tuning as a defense.

**Course:** EEP 595 – Privacy-Preserving Machine Learning, UW Spring 2026
**Author:** Jie Yao

---

## Research Question

When a PII redaction model trained on general-domain text is deployed on clinical notes, adversaries can craft inputs that reliably evade detection. Does fine-tuning the model on clinical data close that vulnerability — and for which attack types?

---

## System Overview

**Target model:** [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) — a 1.5B parameter sparse mixture-of-experts token classifier (50M active per inference) with BIOES tagging and a constrained Viterbi decoder. Detects 8 PII categories: `private_person`, `private_address`, `private_email`, `private_phone`, `private_url`, `private_date`, `account_number`, `secret`.

**Attack taxonomy:**

| Category | Attacks | Expected behavior post fine-tuning |
|---|---|---|
| Character-level | Homoglyph substitution (Latin → Cyrillic), zero-width space injection, thin-space injection | **Persist** — operate below the tokenizer |
| Domain-level | Clinical abbreviations ("J. Williams"), honorific prepending ("Dr.", "Pt."), field-format embedding ("Attending: \<name\>") | **Neutralized** — exploited distribution shift is corrected by fine-tuning |

---

## Project Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Baseline audit — F1 on general vs. clinical domain | ✅ Complete |
| 2 | Adversarial attack suite — evasion rates per attack × entity type | 🔄 In progress |
| 3 | LoRA fine-tuning on MIMIC-IV-Note discharge summaries | ⏳ Upcoming |
| 4 | Re-evaluation — attack evasion before vs. after fine-tuning | ⏳ Upcoming |

---

## Phase 1 Results

Evaluated on 5,000 AI4Privacy validation examples and 500 MIMIC-IV discharge summaries (Hyak RTX 6000 GPU).

**General domain (AI4Privacy) — span-level F1:**

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| `private_person` | 0.187 | 0.578 | 0.282 |
| `private_address` | 0.355 | 0.467 | 0.403 |
| `private_email` | 0.138 | 0.188 | 0.159 |
| `account_number` | 0.028 | 0.505 | 0.054 |
| `secret` | 0.083 | 0.188 | 0.115 |
| **micro avg** | **0.119** | **0.404** | **0.184** |

**Clinical domain (MIMIC) — detection statistics:**
- 500 discharge notes → 530 entities detected → **1.06 entities/note**
- Real unredacted clinical notes typically contain 5–15 PII spans; the low detection rate on already-de-identified MIMIC text confirms the distribution shift hypothesis.

Full evaluation log: [logs/baseline\_eval\_phase1.out](https://github.com/jieyao24/medical-pii-robustness/blob/main/logs/baseline_eval_phase1.out)

---

## Repository Structure

```
medical-pii-robustness/
├── src/
│   ├── baseline_eval.py        # Phase 1: F1 evaluation on AI4Privacy + MIMIC
│   └── adversarial_attacks.py  # Phase 2: 6-attack evasion suite
├── scripts/
│   └── slurm_baseline_eval.sh  # Hyak SLURM job script (ckpt partition, GPU)
├── results/
│   └── baseline_eval.json      # Phase 1 results (F1 + MIMIC detection stats)
├── logs/
│   └── baseline_eval_phase1.out
└── docs/
    ├── Part2_ProgressReport_JieYao.md
    └── EEP509_Project_Proposal&Plan_JieYao.pdf
```

---

## Setup

**Environment:** Python 3.10 on UW Hyak HPC.

```bash
conda create -n pii-robustness python=3.10
conda activate pii-robustness
pip install transformers datasets torch peft evaluate seqeval scikit-learn pandas
```

**Data:**
- AI4Privacy: downloaded automatically via `datasets.load_dataset("ai4privacy/pii-masking-400k")`
- MIMIC-IV-Note 2.2: requires a [PhysioNet Data Use Agreement](https://physionet.org/content/mimic-iv-note/2.2/). Store locally; **never upload raw MIMIC files.**

---

## Running on Hyak

**Phase 1 — Baseline evaluation:**
```bash
# Edit scripts/slurm_baseline_eval.sh: set --account and MIMIC path
sbatch scripts/slurm_baseline_eval.sh
```

**Phase 2 — Adversarial attacks (CPU/single GPU):**
```bash
conda run -n pii-robustness python src/adversarial_attacks.py \
    --max-samples 500 --device 0 --output results/attack_eval.json
```

---

## Key References

- OpenAI. (2025). *OpenAI Privacy Filter Model Card.* `openai/privacy-filter`, HuggingFace.
- Johnson et al. (2023). *MIMIC-IV-Note: Deidentified free-text clinical notes.* PhysioNet v2.2.
- Priva-AI. (2023). *AI4Privacy: Large Language Model PII Masking Dataset.* HuggingFace.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- Boucher et al. (2022). *Bad Characters: Imperceptible NLP Attacks.* IEEE S&P.
