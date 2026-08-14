#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${SAM3_PYTHON:-"$REPO_DIR/.venv-sam3/bin/python"}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "SAM 3 environment not found. Run: $SCRIPT_DIR/setup_sam3.sh" >&2
  exit 1
fi

export PYTHONPATH="$REPO_DIR/pipeline/src:$REPO_DIR/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export SAM3_DEVICE=${SAM3_DEVICE:-auto}
export SAM3_PRECISION=${SAM3_PRECISION:-auto}
export SAM3_HOST=${SAM3_HOST:-127.0.0.1}
export SAM3_PORT=${SAM3_PORT:-8001}

exec "$PYTHON_BIN" -m sam3_runtime.server
