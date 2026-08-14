#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
VENV_DIR="$REPO_DIR/.venv-sam3"

if [ -n "${SAM3_BOOTSTRAP_PYTHON:-}" ]; then
  BOOTSTRAP_PYTHON=$SAM3_BOOTSTRAP_PYTHON
elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
  BOOTSTRAP_PYTHON="$REPO_DIR/.venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON=python3.12
elif command -v python3.11 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON=python3.11
else
  echo "SAM 3 requires Python 3.11 or 3.12 (NumPy <2 has no Python 3.13 arm64 wheel)." >&2
  exit 1
fi

"$BOOTSTRAP_PYTHON" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "SAM 3 requires Python 3.11 or 3.12"'
"$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
  "torch>=2.5" \
  "torchvision>=0.20" \
  "fastapi>=0.115" \
  "uvicorn>=0.30" \
  "python-multipart>=0.0.9" \
  "psutil>=5.9" \
  "opencv-python-headless>=4.10"
"$VENV_DIR/bin/python" -m pip install -e "$REPO_DIR/vendor/sam3"

echo "SAM 3 runtime installed in $VENV_DIR"
echo "Place sam3.pt at $REPO_DIR/pipeline/models/sam3/sam3.pt before starting."
