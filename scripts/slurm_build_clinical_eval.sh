#!/bin/bash
#SBATCH --job-name=pii-inject
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/build_clinical_eval_%j.out
#SBATCH --error=logs/build_clinical_eval_%j.err

# CPU-only job — no GPU needed for this step

set -e

PROJECT_DIR=/mmfs1/gscratch/scrubbed/jieyao24/Projects/medical-pii-robustness
cd "$PROJECT_DIR"
mkdir -p logs data

export HF_HOME=/mmfs1/gscratch/scrubbed/jieyao24/huggingface-cache
export PYTHONUNBUFFERED=1

source /mmfs1/home/jieyao24/miniconda3/etc/profile.d/conda.sh
conda activate pii-robustness

python src/build_clinical_eval.py \
    --mimic-csv /mmfs1/gscratch/scrubbed/jieyao24/data/mimic-iv-note/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz \
    --max-notes 22000 \
    --pool-size 10000 \
    --output data/mimic_pii_injected.jsonl
