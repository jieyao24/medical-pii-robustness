# Medical PII Robustness Project

## Project Summary
Investigating adversarial robustness of PII detection systems in clinical NLP.
Core question: does distribution shift (general → clinical domain) create 
exploitable vulnerabilities in privacy filters, and can LoRA fine-tuning defend against it?

## Course Context
University of Washington, EEP 595: Privacy-Preserving Machine Learning, Spring 2026
Instructor: Tamara Bonaci
Individual project (solo)

## Project Requirements Documents
- `docs/Project_part1.pdf` — Phase 1 requirements
- `docs/Project_part2.pdf` — Phase 2 requirements
- `docs/Project_part3.pdf` — Phase 3 requirements
- `docs/Proposal&Plan_JieYao.docx` — submitted Phase 1 proposal
- `docs/Part2_ProgressReport_JieYao.md` — submitted Phase 2 progress report

## Target Model
`openai/privacy-filter` (HuggingFace)
- Bidirectional token classifier with BIOES tagging + constrained Viterbi decoder
- Sparse MoE FFN: 128 experts, top-4 routing; 1.5B total params, ~50M active
- 8 entity types: account_number, private_address, private_email, private_person,
  private_phone, private_url, private_date, secret (33 output classes with BIOES)
- 128k token context window; based on a converted autoregressive checkpoint (gpt-oss lineage)

## Datasets
- AI4Privacy (HuggingFace, public) — ground-truth PII labeled corpus
- MIMIC-IV-Note 2.2 (PhysioNet, Hyak local only — never upload raw files under any circumstances)
  Download command: wget -r -N -c -np --user jieyao --ask-password https://physionet.org/files/mimic-iv-note/2.2/
  Actual location on Hyak: /mmfs1/gscratch/scrubbed/jieyao24/data/mimic-iv-note/physionet.org/files/mimic-iv-note/2.2/
  Files (all checksums verified ✓):
    note/discharge.csv.gz         — 1.1 GB, ~330k discharge summaries
    note/discharge_detail.csv.gz  — 771 KB
    note/radiology.csv.gz         — 746 MB
    note/radiology_detail.csv.gz  — 38 MB
  NOTE: there is also a partial/incomplete copy at Projects/.../data/physionet.org/ (45 MB, unused, safe to delete)

## HuggingFace Repo
Not used by default. May be used later for derived artifacts (preprocessed outputs, LoRA
checkpoints, evaluation results) only if needed. Raw MIMIC never uploaded under any circumstances.

## GitHub Repo
medical-pii-robustness
Code and configs only. No data files ever.

## Course Deliverable Schedule
| Part | Deliverable | Due |
|------|-------------|-----|
| 1 | Proposal (2p) + Plan (1p) | April 25, 2026 ✓ submitted |
| 2 | 2-page progress report | May 16, 2026 |
| 3 | Draft report (peer-reviewed, ~4-6p solo) | ~June 5, 2026 (end of Week 9) |
| 4 | Final presentation (8–10 min, demo encouraged) + revised report | ~June 13, 2026 (final week) |

Note: Part 3 graded equally on (a) your draft and (b) quality of peer reviews you write for others.
Note: Final presentation demo is strongly encouraged — build with demonstrability in mind.

## Four Research Phases (Technical)
1. Baseline characterization — Privacy Filter F1 on clinical vs. general domain
2. Adversarial attacks — exploiting distribution shift weaknesses
3. LoRA fine-tuning on clinical prose as defense
4. Re-evaluation of attacks post fine-tuning

## Attack Types (Key Distinction)
- Domain-level attacks: clinical abbreviations, syntax obfuscation
  → Fine-tuning expected to DEFEND against these
- Character-level attacks: unicode homoglyphs, whitespace injection
  → Fine-tuning expected to FAIL against these (central finding)

## Privacy-Utility Tradeoff (Key Research Dimension)
- Run fine-tuned model on held-out general-domain data to quantify utility degradation
- Plot 2D tradeoff curve across LoRA ablation configurations
- Catastrophic forgetting is a central research tension, not a footnote
- Ablation infrastructure from Phase 3 supports most of this with minimal extra code

## Key References
- Boucher et al. 2022, "Bad Characters" — directly relevant, must cite
- AI4Privacy dataset paper
- Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"

## Compute
UW Hyak HPC (primary), possibly Tillicum later
Fine-tuning via LoRA (parameter-efficient, runs on a single GPU)
Attack framework: TextAttack

## Development Environment
Claude Code in VS Code (primary IDE)
Conda environment: pii-robustness (Python 3.10)
Synced via GitHub (code only); large artifacts stay on Hyak or HuggingFace only if needed

## Part 2 Technical Tasks (Phase 1: Baseline) — ALL DONE ✓
- [x] Conda env pii-robustness (Python 3.10) created and configured
- [x] AI4Privacy downloaded from HuggingFace
- [x] MIMIC-IV-Note 2.2 downloaded and checksums verified
- [x] openai/privacy-filter downloaded and cached on Hyak
- [x] Baseline eval script written and executed (src/baseline_eval.py)
- [x] F1 table: general domain (AI4Privacy) vs clinical domain (MIMIC) — results/baseline_eval.json

## Part 3 Technical Tasks (Phases 2–4: Attacks + LoRA + Re-evaluation)

### Phase 2: Adversarial Attacks on Baseline Model
- [x] Attack framework implemented — src/adversarial_attacks.py (6 attacks)
- [x] SLURM script created — scripts/slurm_attack_eval.sh
- [x] Attack eval complete — results/attack_eval.json
      Key results (2000 AI4Privacy samples):
        whitespace (char-level): private_person 96.5%, private_address 89.5%  ← nearly perfect
        zwsp       (char-level): account_number 75.2%, private_email 65.9%
        homoglyph  (char-level): private_address 17.4%, private_email 11.4%
        abbreviation (domain):  private_person 10.3%
        honorific    (domain):  private_person 7.0%
        field_format (domain):  private_address 26.4%, private_person 7.0%

### Phase 2.5: PII-Injected MIMIC Evaluation Set
- [x] Write src/build_clinical_eval.py
      Detects ___ placeholders by context (Name:, Attending:, Dr., Date of Birth:, etc.)
      Fixed label mapping bug: AI4Privacy uses GIVENNAME/SURNAME/DATEOFBIRTH/ACCOUNTNUM/TELEPHONENUM
- [x] Run and produce data/mimic_pii_injected.jsonl (gitignored, Hyak only)
      22,000 notes, labeled entities present; test split: notes 21000–22000 (1k held-out)
      Types in test split: private_person 2640 | private_date 909 | account_number 275

### Phase 3: LoRA Fine-Tuning
- [x] Write src/lora_finetune.py (attention-only LoRA: q_proj, v_proj)
- [x] Write scripts/slurm_lora_finetune.sh
- [x] Run 3 jobs: rank=8 fixed × train_size ∈ {1k, 5k, 20k} — ALL COMPLETE ✓
      Results (clinical test loss / final AI4Privacy F1):
        1k:  loss=0.00121  AI4P F1=0.123  (baseline 0.184 → -33% forgetting)
        5k:  loss=0.00026  AI4P F1=0.119  (-35% forgetting)
        20k: loss=0.00013  AI4P F1=0.118  (-36% forgetting)
      Key findings:
        - More data → 10x better clinical fit (loss 0.00121 → 0.00013)
        - Forgetting is small and plateaus: only ~3% extra forgetting from 1k→20k
        - Forgetting stops after epoch 1 in all runs — does not compound with more epochs
- [x] Catastrophic forgetting tracked at each epoch via ForgettingCallback
- [x] Adapters saved to results/checkpoints/lora_r8_n{1000,5000,20000}/adapter/ (gitignored)
      Optional future: add rank=4 and rank=16 at 20k for rank sensitivity

### Phase 3.5: Clinical F1 Evaluation (Baseline + All LoRA Adapters)
- [x] Write src/clinical_eval.py — span-level F1 (exact + overlap) on PII-injected MIMIC test set
- [x] Write scripts/slurm_clinical_eval.sh — 4 parallel jobs (baseline + 3 LoRA adapters)
- [x] Run all 4 jobs — ALL COMPLETE ✓ (jobs 35534858–861)
      Results (test set: notes 21000–22000, 3824 gold spans):

      | Model       | Exact F1 | Overlap P | Overlap R | Overlap F1 |
      |-------------|----------|-----------|-----------|------------|
      | Baseline    | 0.189    | 0.557     | 0.977     | 0.709      |
      | LoRA 1k     | 0.215    | 0.514     | 0.992     | 0.678      |
      | LoRA 5k     | 0.213    | 0.510     | 0.996     | 0.675      |
      | LoRA 20k    | 0.214    | 0.511     | 0.996     | 0.675      |

      Key findings:
        - Fine-tuning raises recall to near-perfect (99.6%) but costs precision
        - Overlap F1 slightly LOWER post-LoRA (0.709 → 0.675): precision trade-off dominates
        - Data scaling plateaus at 1k notes — 5k and 20k add nothing meaningful
        - Exact F1 improves modestly: 0.189 → 0.215 (better boundary alignment)

### Phase 4: Re-Evaluate Attacks on Fine-Tuned Model
- [x] Adapted src/adversarial_attacks.py with --adapter flag
- [x] Write scripts/slurm_attack_eval_lora.sh (3 parallel jobs)
- [x] Run all 6 attacks on fine-tuned models — ALL COMPLETE ✓ (jobs 35535187–189)
      Evasion rates (weighted average):

      | Attack       | Type      | Baseline | LoRA 1k | LoRA 5k | LoRA 20k | Δ(20k) |
      |--------------|-----------|----------|---------|---------|----------|---------|
      | homoglyph    | char      | 13.4%    | 14.7%   | 15.7%   | 22.5%    | +9.1%  |
      | zwsp         | char      | 52.5%    | 68.3%   | 67.7%   | 78.5%    | +26.0% |
      | whitespace   | char      | 91.5%    | 92.7%   | 90.8%   | 92.7%    | +1.2%  |
      | abbreviation | domain    | 10.3%    | 10.9%   | 10.9%   | 10.8%    | +0.5%  |
      | honorific    | domain    | 7.0%     | 7.2%    | 6.8%    | 6.6%     | -0.5%  |
      | field_format | domain    | 20.9%    | 27.7%   | 25.1%   | 27.3%    | +6.4%  |

      **CRITICAL FINDING — counter to hypothesis:**
        LoRA INCREASES evasion on char-level attacks (+1.2% to +26%)
        LoRA provides minimal defense on domain-level attacks (±0.5%)
        → Fine-tuning on clinical text does NOT defend against adversarial perturbations
        → Catastrophic forgetting in adversarial robustness domain, not just utility
- [x] Plot privacy-utility tradeoff curve and evasion comparison

### Report
- [ ] Write draft report (~4–6 pages) — due ~June 5, 2026
- [ ] Write peer reviews for classmates

## Current Status (as of 2026-05-25)

### Completed
- Part 1 (proposal + plan) submitted ✓ — April 25, 2026
- Part 2 (progress report) submitted ✓ — May 16, 2026
- MIMIC-IV-Note 2.2 downloaded and checksums verified ✓
- Baseline evaluation run ✓ — results/baseline_eval.json
  → AI4Privacy general domain: micro F1 = 0.184 (P=0.119, R=0.404)
- Adversarial attacks on baseline model ✓ — results/attack_eval.json
  → whitespace: 96.5% evasion (private_person); zwsp: 75.2% (account_number)
  → char-level attacks far more effective than domain-level attacks
- PII-injected MIMIC eval set built ✓ — data/mimic_pii_injected.jsonl (22k notes, gitignored)
- LoRA fine-tuning ablation ✓ — 3 adapters at results/checkpoints/lora_r8_n{1000,5000,20000}/adapter/
  → AI4P F1 after fine-tuning: 0.123 / 0.119 / 0.118 (mild forgetting, plateaus)
- Clinical F1 evaluation ✓ — results/clinical_eval_{baseline,lora_r8_n*}.json
  → Baseline overlap F1 = 0.709; LoRA 20k overlap F1 = 0.675 (recall↑, precision↓)

### Not Started (remaining work)
1. Phase 4: Re-run 6 adversarial attacks on fine-tuned model (lora_r8_n20000 recommended)
   → Expected: domain-level attacks defended; char-level attacks persist
2. Privacy-utility tradeoff plot (2D: clinical F1 vs AI4P F1 across ablation configs)
3. Draft report (~4–6 pages) — due ~June 5, 2026
4. Peer reviews for classmates

## Critical Constraints
- NEVER upload raw MIMIC files to GitHub, HuggingFace, or anywhere else (PhysioNet DUA)
- Raw MIMIC lives on Hyak local storage only (data/ directory is gitignored)
- HuggingFace is optional; only derived/preprocessed outputs may ever go there
- *.jsonl, *.parquet, *.csv.gz, *.bin, *.safetensors are all gitignored (see .gitignore)