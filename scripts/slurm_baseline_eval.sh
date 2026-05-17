#!/bin/bash
#SBATCH --job-name=pii-baseline
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=z3005,z3006
#SBATCH --time=2:00:00
#SBATCH --output=logs/baseline_eval_%j.out
#SBATCH --error=logs/baseline_eval_%j.err

set -e

PROJECT_DIR=/mmfs1/gscratch/scrubbed/jieyao24/Projects/medical-pii-robustness
cd "$PROJECT_DIR"
mkdir -p logs results

# Point HuggingFace cache to scrubbed (avoid filling home quota)
export HF_HOME=/mmfs1/gscratch/scrubbed/jieyao24/huggingface-cache
export PYTHONUNBUFFERED=1

source /mmfs1/home/jieyao24/miniconda3/etc/profile.d/conda.sh
conda activate pii-robustness

python src/baseline_eval.py \
    --mimic-dir /mmfs1/gscratch/scrubbed/jieyao24/data/mimic-iv-note/physionet.org/files/mimic-iv-note/2.2 \
    --max-ai4privacy 5000 \
    --max-mimic 500 \
    --device 0 \
    --output results/baseline_eval.json
