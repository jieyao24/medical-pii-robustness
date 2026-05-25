#!/bin/bash
# LoRA fine-tuning ablation launcher.
#
# Submits one SLURM job per (rank, train_size) combination.
# Usage:  bash scripts/slurm_lora_finetune.sh
#
# To run a single config manually:
#   sbatch --export=RANK=8,TRAIN_SIZE=5000 scripts/slurm_lora_finetune.sh

#SBATCH --job-name=pii-lora
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=z3005,z3006
#SBATCH --time=8:00:00
#SBATCH --output=logs/lora_%x_r%x_%j.out
#SBATCH --error=logs/lora_%x_r%x_%j.err

# ── If run as a script (not sbatch), submit rank=8 across 3 train sizes ─────
# Full 9-run ablation is overkill for a 4-6p report. Fix rank=8 (standard
# LoRA default), vary train size to show clinical domain scaling effect.
# To add rank ablation later: manually sbatch with RANK=4 or RANK=16.
if [ -z "$SLURM_JOB_ID" ]; then
    RANK=8
    TRAIN_SIZES=(1000 5000 20000)
    for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do
        sbatch \
            --job-name="lora_r${RANK}_n${TRAIN_SIZE}" \
            --output="logs/lora_r${RANK}_n${TRAIN_SIZE}_%j.out" \
            --error="logs/lora_r${RANK}_n${TRAIN_SIZE}_%j.err" \
            --export="RANK=${RANK},TRAIN_SIZE=${TRAIN_SIZE}" \
            scripts/slurm_lora_finetune.sh
        echo "Submitted: rank=${RANK} train_size=${TRAIN_SIZE}"
    done
    exit 0
fi

# ── Actual SLURM job body ────────────────────────────────────────────────────
set -e

PROJECT_DIR=/mmfs1/gscratch/scrubbed/jieyao24/Projects/medical-pii-robustness
cd "$PROJECT_DIR"
mkdir -p logs results/checkpoints

export HF_HOME=/mmfs1/gscratch/scrubbed/jieyao24/huggingface-cache
export PYTHONUNBUFFERED=1

source /mmfs1/home/jieyao24/miniconda3/etc/profile.d/conda.sh
conda activate pii-robustness

echo "Starting LoRA fine-tuning: rank=${RANK}, train_size=${TRAIN_SIZE}"

python src/lora_finetune.py \
    --lora-rank "${RANK}" \
    --train-size "${TRAIN_SIZE}" \
    --output-dir "results/checkpoints/lora_r${RANK}_n${TRAIN_SIZE}" \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-4 \
    --device 0 \
    --forgetting-samples 2000
