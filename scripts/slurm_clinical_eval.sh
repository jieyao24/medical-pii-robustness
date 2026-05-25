#!/bin/bash
# Clinical eval launcher — submits 4 parallel jobs (baseline + 3 LoRA adapters).
# Usage: bash scripts/slurm_clinical_eval.sh

#SBATCH --job-name=pii-ceval
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=z3005,z3006
#SBATCH --time=1:00:00
#SBATCH --output=logs/clinical_eval_%x_%j.out
#SBATCH --error=logs/clinical_eval_%x_%j.err

# ── If called as a script, submit 4 jobs in parallel ────────────────────────
if [ -z "$SLURM_JOB_ID" ]; then
    declare -A CONFIGS=(
        ["baseline"]=""
        ["lora_r8_n1000"]="results/checkpoints/lora_r8_n1000/adapter"
        ["lora_r8_n5000"]="results/checkpoints/lora_r8_n5000/adapter"
        ["lora_r8_n20000"]="results/checkpoints/lora_r8_n20000/adapter"
    )
    for NAME in "${!CONFIGS[@]}"; do
        ADAPTER="${CONFIGS[$NAME]}"
        sbatch \
            --job-name="ceval_${NAME}" \
            --output="logs/clinical_eval_${NAME}_%j.out" \
            --error="logs/clinical_eval_${NAME}_%j.err" \
            --export="EVAL_NAME=${NAME},ADAPTER=${ADAPTER}" \
            scripts/slurm_clinical_eval.sh
        echo "Submitted: ${NAME}"
    done
    exit 0
fi

# ── Actual SLURM job body ────────────────────────────────────────────────────
set -e

PROJECT_DIR=/mmfs1/gscratch/scrubbed/jieyao24/Projects/medical-pii-robustness
cd "$PROJECT_DIR"
mkdir -p logs results

export HF_HOME=/mmfs1/gscratch/scrubbed/jieyao24/huggingface-cache
export PYTHONUNBUFFERED=1

source /mmfs1/home/jieyao24/miniconda3/etc/profile.d/conda.sh
conda activate pii-robustness

echo "Running clinical eval: ${EVAL_NAME}"

if [ -z "$ADAPTER" ]; then
    python src/clinical_eval.py \
        --data data/mimic_pii_injected.jsonl \
        --output "results/clinical_eval_${EVAL_NAME}.json" \
        --device 0
else
    python src/clinical_eval.py \
        --data data/mimic_pii_injected.jsonl \
        --adapter "$ADAPTER" \
        --output "results/clinical_eval_${EVAL_NAME}.json" \
        --device 0
fi

echo "Done: ${EVAL_NAME}"
