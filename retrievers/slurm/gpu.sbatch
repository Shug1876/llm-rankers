#!/usr/bin/env bash
# SPLADE / E5 / ColBERT job: one L40S for encoding, CPUs for PISA and numpy search. Arguments go to retrievers/run.py.
#
#   sbatch retrievers/slurm/gpu.sbatch --retriever e5 --dataset beir
#   sbatch --partition=gpu-h100 --gres=gpu:h100:1 retrievers/slurm/gpu.sbatch --retriever colbert --dataset miracl/en
#   bash   retrievers/slurm/gpu.sbatch --retriever splade --dataset beir/scifact    # on a node you already hold
#SBATCH --job-name=rb-gpu
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=retrievers/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p retrievers/logs

VENV=${VENV:-/mnt/scratch/users/3148123l/venvs/retrievers}
export HF_HOME=${HF_HOME:-/mnt/scratch/users/3148123l/hf_home}
export IR_DATASETS_HOME=${IR_DATASETS_HOME:-/mnt/scratch/users/3148123l/ir_datasets}  # corpora are large; home has a 250G quota

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-$(nproc)} gpu=${CUDA_VISIBLE_DEVICES:-none} args=$*"
"$VENV/bin/python" retrievers/run.py "$@"
