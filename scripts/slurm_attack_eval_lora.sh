#!/bin/bash
# Attack eval on LoRA-adapted models — submits 3 parallel jobs (one per adapter).
# Usage: bash scripts/slurm_attack_eval_lora.sh
#
# Evasion rates are compared against baseline results/attack_eval.json
# to test whether fine-tuning defended domain-level but not char-level attacks.

#SBATCH --job-name=pii-attack-lora
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=z3005,z3006
#SBATCH --time=4:00:00
#SBATCH --output=logs/attack_lora_%x_%j.out
#SBATCH --error=logs/attack_lora_%x_%j.err

# ── If called as a script, submit 3 jobs in parallel ────────────────────────
if [ -z "$SLURM_JOB_ID" ]; then
    declare -A ADAPTERS=(
        ["lora_r8_n1000"]="results/checkpoints/lora_r8_n1000/adapter"
        ["lora_r8_n5000"]="results/checkpoints/lora_r8_n5000/adapter"
        ["lora_r8_n20000"]="results/checkpoints/lora_r8_n20000/adapter"
    )
    for NAME in "${!ADAPTERS[@]}"; do
        ADAPTER="${ADAPTERS[$NAME]}"
        sbatch \
            --job-name="attack_${NAME}" \
            --output="logs/attack_lora_${NAME}_%j.out" \
            --error="logs/attack_lora_${NAME}_%j.err" \
            --export="EVAL_NAME=${NAME},ADAPTER=${ADAPTER}" \
            scripts/slurm_attack_eval_lora.sh
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

echo "Running attack eval: ${EVAL_NAME}  adapter=${ADAPTER}"

python src/adversarial_attacks.py \
    --adapter "$ADAPTER" \
    --max-samples 2000 \
    --device 0 \
    --output "results/attack_eval_${EVAL_NAME}.json"

echo "Done: ${EVAL_NAME}"
