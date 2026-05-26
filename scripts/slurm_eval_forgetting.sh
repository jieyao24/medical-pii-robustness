#!/bin/bash
#SBATCH --job-name=pii-forgetting
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=z3005,z3006
#SBATCH --time=1:00:00
#SBATCH --output=logs/eval_forgetting_%j.out
#SBATCH --error=logs/eval_forgetting_%j.err

# Post-hoc forgetting eval — evaluates all 3 saved LoRA adapters on AI4Privacy
# with the corrected label map. Writes forgetting_history_corrected.json for each.
# Run AFTER the baseline_eval rerun (ensures corrected baseline F1 is available).
#
# Usage: sbatch scripts/slurm_eval_forgetting.sh

set -e

PROJECT_DIR=/mmfs1/gscratch/scrubbed/jieyao24/Projects/medical-pii-robustness
cd "$PROJECT_DIR"
mkdir -p logs

export HF_HOME=/mmfs1/gscratch/scrubbed/jieyao24/huggingface-cache
export PYTHONUNBUFFERED=1

source /mmfs1/home/jieyao24/miniconda3/etc/profile.d/conda.sh
conda activate pii-robustness

python src/eval_forgetting.py \
    --device 0 \
    --max-samples 2000

echo "Forgetting eval complete."
