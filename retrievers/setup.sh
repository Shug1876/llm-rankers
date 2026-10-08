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

if ! command -v uv >/dev/null; then
  echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
uv pip install --python "$VENV/bin/python" -r retrievers/requirements.txt

"$VENV/bin/python" - <<'EOF'
import importlib
for name in ["pyterrier", "pyterrier_pisa", "pyterrier_splade", "pyterrier_dr", "pylate", "pyterrier_pylate",
             "ir_datasets", "ir_measures", "datasets", "torch"]:
    mod = importlib.import_module(name)
    print(f"{name:18s} {getattr(mod, '__version__', '?')}")
import torch
print("cuda available:", torch.cuda.is_available(), "(expected False on the login node)")
EOF
mkdir -p retrievers/logs   # sbatch needs the --output directory to exist at submit time
echo "venv ready: $VENV"
