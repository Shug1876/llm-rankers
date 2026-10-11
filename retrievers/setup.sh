#!/usr/bin/env bash
# Create the uv venv for the retriever benchmark and check that every backend imports.
# Run once on the login node (it has internet through the proxy), from anywhere:
#
#   bash retrievers/setup.sh
#   VENV=/path/to/other/venv bash retrievers/setup.sh     # different location
#
# PISA and the Hugging Face models need no Java; nothing here starts a JVM.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${VENV:-/mnt/scratch/users/3148123l/venvs/retrievers}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
# PyPI's default torch wheel is built for a new CUDA (12.8/13.0) and refuses to start on an older driver
# ("NVIDIA driver on your system is too old (found version 12020)"). Pick the wheel index to match the cluster's
# driver (`nvidia-smi` shows its max CUDA version): cu121 runs on any driver >= 12.1, e.g. the 12.2 one here.
# Set TORCH_INDEX= (empty) to use PyPI's default build instead.
TORCH_INDEX=${TORCH_INDEX-https://download.pytorch.org/whl/cu121}

if ! command -v uv >/dev/null; then
  echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
if [ -n "$TORCH_INDEX" ]; then
  # first, so the requirements below see torch already satisfied and do not pull PyPI's newer-CUDA build over it;
  # this also replaces a wrong-CUDA torch left in an existing venv
  uv pip install --python "$VENV/bin/python" torch --index-url "$TORCH_INDEX"
fi
uv pip install --python "$VENV/bin/python" -r retrievers/requirements.txt
# pyterrier-pylate is only on GitHub and only ColBERT needs it: a pod that cannot reach github.com still gets
# a venv that runs BM25, SPLADE and E5 (one failing requirement would otherwise abort the whole install).
COLBERT=pyterrier_pylate
if ! uv pip install --python "$VENV/bin/python" "pyterrier-pylate @ git+https://github.com/lightonai/pyterrier-pylate"; then
  echo "WARNING: pyterrier-pylate not installed (github.com unreachable?); --retriever colbert will not run" >&2
  COLBERT=
fi

COLBERT=$COLBERT "$VENV/bin/python" - <<'EOF'
import importlib, os
for name in ["pyterrier", "pyterrier_pisa", "pyterrier_splade", "pyterrier_dr", "pylate", os.environ["COLBERT"],
             "ir_datasets", "ir_measures", "datasets", "torch"]:
    if not name:
        continue
    mod = importlib.import_module(name)
    print(f"{name:18s} {getattr(mod, '__version__', '?')}")
import torch
print("torch built for CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "(expected False on the login node)")
EOF
mkdir -p retrievers/logs   # sbatch needs the --output directory to exist at submit time
echo "venv ready: $VENV"
