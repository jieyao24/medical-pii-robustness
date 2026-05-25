#!/bin/bash
#SBATCH --job-name=pii-attack
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=z3005,z3006
#SBATCH --time=3:00:00
#SBATCH --output=logs/attack_eval_%j.out
#SBATCH --error=logs/attack_eval_%j.err

set -e

PROJECT_DIR=/mmfs1/gscratch/scrubbed/jieyao24/Projects/medical-pii-robustness
cd "$PROJECT_DIR"
mkdir -p logs results

export HF_HOME=/mmfs1/gscratch/scrubbed/jieyao24/huggingface-cache
export PYTHONUNBUFFERED=1

source /mmfs1/home/jieyao24/miniconda3/etc/profile.d/conda.sh
conda activate pii-robustness

python src/adversarial_attacks.py \
    --max-samples 2000 \
    --device 0 \
    --output results/attack_eval.json
