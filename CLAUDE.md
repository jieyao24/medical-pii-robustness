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
- `docs/Whole_project.pdf` — full project requirements from instructor (all phases)
- `docs/Proposal&Plan_JieYao.docx` — submitted Phase 1 proposal

## Target Model
OpenAI Privacy Filter (sparse MoE PII detection model, HuggingFace)

## Datasets
- AI4Privacy (HuggingFace, public) — ground-truth PII labeled corpus
- MIMIC-IV-Note 2.2 (PhysioNet, local only — never upload raw files under any circumstances)
- Derived: PII-injected clinical dataset (HuggingFace private repo)

## HuggingFace Repo
Private dataset repo: yourname/eep595-clinical-pii
Stores: preprocessed MIMIC sentences, PII-injected clinical dataset,
        evaluation results, training logs, LoRA checkpoints

## GitHub Repo
medical-pii-robustness
Code and configs only. No data files ever.

## Four Research Phases
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
Synced via GitHub; large artifacts via HuggingFace private repo

## Current Status
- Phase 1 proposal submitted
- MIMIC-IV-Note 2.2 access confirmed via PhysioNet DUA
- Setting up data pipeline and baseline evaluation (Part 2 progress report in progress)

## Critical Constraints
- NEVER upload raw MIMIC files to GitHub, HuggingFace, or anywhere else (PhysioNet DUA)
- Only derived and preprocessed outputs may be pushed to HuggingFace
- Raw MIMIC lives on Hyak local storage only