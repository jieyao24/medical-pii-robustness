# EEP 595 – Course Project Part 2: Progress Report
**Jie Yao | May 16, 2026**

**Code repository:** [github.com/jieyao24/medical-pii-robustness](https://github.com/jieyao24/medical-pii-robustness)

---

## 1. Progress and Execution

### What Has Been Accomplished

Progress this phase covers three areas: system and data characterization, baseline evaluation implementation and execution, and adversarial attack framework design.

**Target system and dataset characterization.**
The victim model, `openai/privacy-filter` (HuggingFace, Apache 2.0), is a bidirectional token classifier using BIOES span tagging with a constrained Viterbi decoder. Its feed-forward layers implement a sparse mixture-of-experts (MoE) architecture with 128 experts and top-4 routing, yielding 1.5B total parameters but only ~50M active per inference. It detects 8 entity types: `account_number`, `private_address`, `private_email`, `private_person`, `private_phone`, `private_url`, `private_date`, and `secret`. Understanding this architecture, especially the MoE routing, was necessary before designing either attacks or a fine-tuning strategy.

MIMIC-IV-Note 2.2 access was confirmed via PhysioNet Data Use Agreement. The full 1.1 GB discharge summary corpus (`discharge.csv.gz`, ~330k notes) has been downloaded to Hyak local storage and will not leave that environment. The AI4Privacy corpus (HuggingFace, 400k+ annotated examples across 50+ PII categories) was used as the general-domain benchmark. Its `privacy_mask` field provides character-offset span annotations, which align naturally with the character-level output of the `transformers` pipeline.

**Baseline evaluation pipeline — implemented and executed.**
A complete baseline evaluation script (`src/baseline_eval.py`) was implemented and run on Hyak HPC (RTX 6000 GPU, `ckpt` partition). The pipeline evaluates `openai/privacy-filter` on both domains and saves per-category results as JSON. Results are described in Section 4.

**Adversarial attack framework — designed and implemented.**
A six-attack evasion suite (`src/adversarial_attacks.py`) has been implemented, covering the full attack taxonomy from the proposal. Each attack perturbs a gold PII span and measures whether the model still detects it. The framework outputs per-attack, per-entity-type evasion rates. Execution is pending SLURM job submission (blocked on evaluation job completion; ready to run immediately).

**Attack taxonomy design.**
The attack surface is partitioned into two categories based on expected interaction with fine-tuning:

- *Character-level attacks* — unicode homoglyph substitution (Latin → Cyrillic look-alikes), zero-width space injection, and thin-space injection. These operate below the tokenizer: the model sees different byte sequences regardless of domain. Expected to remain effective post fine-tuning. This is the central finding the project is designed to demonstrate.
- *Domain-level attacks* — clinical abbreviation formatting (e.g., "John Williams" → "J. Williams"), honorific prepending ("Dr.", "Pt.", "Attending:"), and structured field embedding ("Patient: \<name\>"). These exploit the model's unfamiliarity with clinical prose and are expected to be neutralized by domain-adaptive fine-tuning.

### Obstacles and Challenges Encountered

The most time-consuming obstacle was iterative schema discovery for the AI4Privacy dataset. The dataset's column names and annotation format differ from its documentation (`ner_tags` → `mbert_bio_labels` → `privacy_mask` with character offsets), requiring multiple SLURM job submissions to diagnose. In retrospect, inspecting dataset features locally before committing to batch jobs would have saved significant time.

A second engineering challenge was GPU compatibility on the Hyak `ckpt` partition. The partition allocates idle nodes from all GPU types, including legacy P100 nodes (CUDA CC 6.0) that are incompatible with the installed PyTorch build. This was resolved by adding a node exclusion directive (`--exclude=z3005,z3006`) to the SLURM script.

A third conceptual challenge remains: applying LoRA to a sparse MoE model. Standard PEFT targets dense attention and MLP layers; with MoE, it is unclear whether to adapt the router, the expert weights, or only the attention projections. This decision will affect both training stability and the expressiveness of the domain adaptation.

### Anticipated Harder Steps and Mitigation Plans

| Challenge | Mitigation |
|---|---|
| MoE LoRA instability | Begin with attention-only LoRA; ablate expert-layer LoRA separately |
| MIMIC PII injection naturalism | Validate injected spans against clinical grammar; spot-check 100 notes |
| Catastrophic forgetting post fine-tuning | Evaluate on held-out AI4Privacy split at every LoRA checkpoint |
| Evasion rate sensitivity to span matching | Use overlap-based matching in addition to exact matching |

---

## 2. Target Model and Datasets

**Model: `openai/privacy-filter`.** A 1.5B parameter sparse MoE token classifier released by OpenAI in April 2025 under Apache 2.0. It uses a bidirectional encoder (GPT-OSS lineage, converted from autoregressive) with BIOES span tagging and a constrained Viterbi decoder over a 128k-token context window. With MoE routing (128 experts, top-4), only ~50M parameters are active per forward pass, making inference feasible on a single consumer GPU. The model was designed as a drop-in sanitization layer for high-throughput data pipelines.

**AI4Privacy (general domain).** A public HuggingFace dataset with 400k+ annotated examples across 50+ PII entity types drawn from financial, legal, and social media text. For evaluation, the 50+ fine-grained labels are mapped to the model's 8 output categories (e.g., `FIRSTNAME`, `LASTNAME` → `private_person`). Annotations are provided as character-offset spans in the `privacy_mask` field. This corpus serves as both the baseline general-domain benchmark and the post-fine-tuning utility measurement set.

**MIMIC-IV-Note 2.2 (clinical domain).** A PhysioNet dataset of 330k+ de-identified clinical notes from Beth Israel Deaconess Medical Center, including discharge summaries, radiology reports, and progress notes. Because MIMIC is already de-identified, it cannot be used directly for F1 evaluation. For Phase 1, it provides a domain characterization: how the model behaves on authentic clinical prose. For Phase 3 (fine-tuning), raw MIMIC discharge summaries are the adaptation corpus. For adversarial evaluation (Phases 2 and 4), AI4Privacy PII entities are injected into MIMIC sentences to create a labeled clinical benchmark.

---

## 3. Baseline Evaluation Methodology

The baseline evaluation runs `openai/privacy-filter` on both domains via the HuggingFace `transformers` pipeline with `aggregation_strategy="simple"`, which returns character-offset entity spans.

**General domain (AI4Privacy):** 5,000 examples are sampled from the validation split. For each example, gold PII spans are extracted from the `privacy_mask` field and mapped to the model's 8 categories. The model's predicted spans are compared to gold spans using **exact span-level matching**: a predicted span counts as a true positive only if it matches a gold span on both character offsets and entity type. Span-level precision, recall, and F1 are computed per entity type and aggregated as micro averages.

**Clinical domain (MIMIC):** Since MIMIC is already de-identified, no gold labels exist. For Phase 1, the model is run on 500 discharge summaries and entity detection statistics (count, rate, type distribution) are recorded. This establishes a behavioral baseline on clinical text before the PII injection evaluation set is constructed.

---

## 4. Results

Full evaluation log: [logs/baseline\_eval\_phase1.out](https://github.com/jieyao24/medical-pii-robustness/blob/main/logs/baseline_eval_phase1.out)

### General Domain — AI4Privacy (5,000 validation examples)

| Entity Type | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `private_person` | 0.187 | 0.578 | 0.282 | 761 |
| `private_address` | 0.355 | 0.467 | 0.403 | 1,760 |
| `private_email` | 0.138 | 0.188 | 0.159 | 848 |
| `account_number` | 0.028 | 0.505 | 0.054 | 190 |
| `secret` | 0.083 | 0.188 | 0.115 | 362 |
| `private_url/phone/date` | 0.000 | 0.000 | 0.000 | 0 |
| **micro avg** | **0.119** | **0.404** | **0.184** | 3,921 |

### Clinical Domain — MIMIC-IV-Note (500 discharge summaries)

| Metric | Value |
|---|---|
| Notes evaluated | 500 |
| Total entities detected | 530 |
| Entities per note | 1.06 |
| `private_person` | 309 (58%) |
| `private_date` | 206 (39%) |
| `account_number` | 13 (2.5%) |
| `private_address` | 2 (0.4%) |

---

## 5. Observations and Interpretation

**Low precision, moderate recall on general domain.** The micro F1 of 0.184 is driven by very low precision (0.119): the model fires on many spans that do not match gold entities, producing a large number of false positives. Recall is more respectable at 0.404, meaning roughly 40% of annotated PII is detected. The exact-span matching metric is strict — a predicted span that is off by even one character counts as both a false positive and a false negative — so true performance is likely somewhat higher. This will be addressed in Phase 2 with overlap-based matching.

**Three categories show zero support.** `private_url`, `private_phone`, and `private_date` have no gold entities in the 5,000-sample validation subset. This is likely a sampling artifact; running on the full validation set (81k examples) would populate these categories.

**Clinical detection rate is strikingly low.** The model detects only 1.06 entities per MIMIC discharge note. Real unredacted clinical notes typically contain 5–15 PII spans (patient names, dates, provider identifiers, addresses). Since MIMIC is already de-identified, the model correctly finds few PII spans, but the distribution of detections (dominated by `private_person` and `private_date`, almost no `private_address`) suggests the model fires on de-identification artifacts (e.g., placeholder dates like "2180-05-07" inserted by the de-identification pipeline) rather than on clinical-style PII. This confirms the distribution shift hypothesis: the model was not trained on clinical prose and its behavior on clinical text is qualitatively different from its behavior on general-domain text.

---

## 6. Relationship with Papers

**Boucher et al. 2022, "Bad Characters: Imperceptible NLP Attacks."** This paper demonstrates that invisible or visually identical Unicode characters — homoglyphs, zero-width spaces, reorderings — reliably fool transformer-based NLP models while remaining invisible to human reviewers, with no model access required. The attack is directly applicable here: a Cyrillic "а" replacing a Latin "a" in a patient name produces a different token ID sequence, which the model may fail to tag as `private_person`. Because the perturbation is sub-tokenizer, fine-tuning on more clinical text cannot fix it — the tokenizer sees a different byte sequence regardless of domain adaptation. A technical gap in the paper is that it evaluates only sequence-level classifiers (sentiment, toxicity). Adapting the attack to span-level PII detection requires redefining attack success as: a perturbation succeeds if a previously detected entity is missed after perturbation. This definition is implemented in `src/adversarial_attacks.py`.

**Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models."** LoRA freezes pre-trained weights and injects trainable low-rank matrices (*r* ≪ *d*) into attention projections. At inference, the matrices are merged with frozen weights at zero latency cost. For this project, LoRA is the only computationally feasible strategy: full fine-tuning of a 1.5B parameter model on Hyak's single-GPU allocation is infeasible, but LoRA at rank 8–16 reduces the trainable parameter count to millions. The key open question is which layers to adapt in a MoE model. The original paper targets dense attention projections; adapting the MoE expert weights may yield richer domain adaptation but risks router instability. The plan is to begin with attention-only LoRA and ablate expert-layer adaptation separately if time allows.

**AI4Privacy Dataset.** The dataset provides 400k+ annotated PII examples across 50+ types. In practice, the dataset's annotation format evolved across versions (token-level BIO tags → character-offset `privacy_mask`), requiring schema inspection before reliable evaluation. The main methodological concern is entity type alignment: AI4Privacy's fine-grained labels (50+) must be collapsed into the model's 8 categories, introducing label merging uncertainty that will be documented as a measurement limitation in the final report.

---

## 7. Plan for Part 3

**Immediate (Week 8):** Run the adversarial attack suite against the baseline model. All six attacks are implemented and ready. The SLURM script (`scripts/slurm_baseline_eval.sh`) will be adapted for attack evaluation. Expected outputs: evasion rate table per attack × entity type, confirming which categories are most vulnerable.

**Week 8–9:** Construct the PII-injected MIMIC evaluation set by inserting AI4Privacy entities into MIMIC discharge sentences. This gives a labeled clinical benchmark for proper F1 evaluation on clinical text — the core comparison table (general-domain F1 vs. clinical-domain F1) that motivates the project.

**Week 9:** LoRA fine-tuning of `openai/privacy-filter` on MIMIC-IV-Note discharge summaries using Hyak H200 nodes. Ablate over LoRA rank (4, 8, 16) and training set size (1k, 5k, 20k notes). Monitor for catastrophic forgetting by evaluating on held-out AI4Privacy split at each checkpoint.

**Week 9–10:** Re-run the full adversarial attack suite against the fine-tuned model. Compare evasion rates before and after fine-tuning, separated by attack category (domain-level vs. character-level). This produces the central result: domain-level attacks are neutralized; character-level attacks persist. Write draft report.
