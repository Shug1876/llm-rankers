#!/usr/bin/env bash
# Serve the open JEV model (autotrust/JEV-9B, a distilled Jev 1.13) with vLLM on one cognition GPU, for
# `run_jev.py --provider local`. Submit from the repo root:
#
#   sbatch jev/serve_jev9b.sbatch
#   cat jev/logs/jev9b.endpoint          # http://<node>:<port>, written when the job starts
#
# 100 passages in one request (listwise --window_size 100) need MAX_MODEL_LEN=32768.
# Start-up takes a few minutes (weights + CUDA graphs); the server is ready when GET <endpoint>/v1/models answers.
# One-time setup (login node, it has internet through the proxy):
#   uv venv --python 3.12 $VENV && uv pip install --python $VENV/bin/python vllm==0.31.0 -r jev/requirements.txt
#   hf download autotrust/JEV-9B --local-dir $JEV_MODEL_DIR --exclude "reports/*" --exclude "vl/*" --exclude "videos/*"
#SBATCH --job-name=jev9b
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=jev/logs/%x-%j.out
set -euo pipefail

VENV=${VENV:-/mnt/scratch/users/3148123l/venvs/jev-vllm}
JEV_MODEL_DIR=${JEV_MODEL_DIR:-/mnt/scratch/users/3148123l/models/JEV-9B}
PORT=${PORT:-$((8000 + SLURM_JOB_ID % 1000))}
ENDPOINT_FILE=${ENDPOINT_FILE:-jev/logs/jev9b.endpoint}   # e.g. jev/logs/jev9b-32k.endpoint for a second server

# The cluster proxy breaks local and node-to-node HTTP; weights are already on disk.
unset http_proxy https_proxy ftp_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY
export HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-/mnt/scratch/users/3148123l/hf_home}

mkdir -p jev/logs
echo "http://$(hostname):$PORT" > "$ENDPOINT_FILE"
echo "job $SLURM_JOB_ID serving $JEV_MODEL_DIR at $(cat "$ENDPOINT_FILE"), time limit $(squeue -h -j "$SLURM_JOB_ID" -o %l)"

exec "$VENV/bin/vllm" serve "$JEV_MODEL_DIR" --served-model-name autotrust/JEV-9B \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision="$JEV_MODEL_DIR/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --enable-prefix-caching --mamba-cache-mode align \
  --host 0.0.0.0 --port "$PORT" ${EXTRA_ARGS:-}
