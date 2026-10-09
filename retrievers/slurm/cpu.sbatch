#!/usr/bin/env bash
# BM25 (PISA) job: CPU only, every core of the allocation. All arguments go to retrievers/run.py.
#
#   sbatch retrievers/slurm/cpu.sbatch --retriever bm25 --dataset beir
#   bash   retrievers/slurm/cpu.sbatch --retriever bm25 --dataset beir/scifact     # same thing, no SLURM
#
# The cluster only has GPU partitions; this job asks for none of the GPUs.
#SBATCH --job-name=rb-cpu
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=retrievers/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p retrievers/logs

VENV=${VENV:-/mnt/scratch/users/3148123l/venvs/retrievers}
export HF_HOME=${HF_HOME:-/mnt/scratch/users/3148123l/hf_home}
export IR_DATASETS_HOME=${IR_DATASETS_HOME:-/mnt/scratch/users/3148123l/ir_datasets}  # corpora are large; home has a 250G quota

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-$(nproc)} args=$*"
"$VENV/bin/python" retrievers/run.py "$@"
