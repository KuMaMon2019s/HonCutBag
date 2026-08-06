#!/usr/bin/env bash

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"
CONDA_ENV="${HONCUT_CONDA_ENV:-honcut}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is required to run the HonCut pipeline." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
python -c "from pipeline.src.utils.deps import check_dependencies; check_dependencies(); print('Dependency check passed')"

cd "${PIPELINE_DIR}/src"
exec python pipeline_runner.py "$@"
