#!/usr/bin/env bash
# Create eye_annotator on Linux without sudo (conda-forge only).
# Usage (from repo root):  bash scripts/setup_annotator_linux.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniforge/Miniconda in your home directory first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx eye_annotator; then
  echo "Updating existing conda env: eye_annotator"
  conda env update -f environment_annotator_linux.yml --prune
else
  echo "Creating conda env: eye_annotator"
  conda env create -f environment_annotator_linux.yml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eye_annotator

echo "Verifying imports..."
python - <<'PY'
import cv2
from PyQt6 import QtCore
import pyqtgraph
import open_ephys.analysis  # noqa: F401
print("OK:", cv2.__version__, QtCore.QT_VERSION_STR)
PY

if [[ -z "${DISPLAY:-}" ]]; then
  echo
  echo "Note: DISPLAY is unset. For a GUI you need X11 forwarding, a desktop session,"
  echo "or:  conda install -c conda-forge xorg-xvfb && xvfb-run -a python -m eye_tracking_system_tools.annotation.block_annotator"
fi

echo
echo "Launch:"
echo "  conda activate eye_annotator"
echo "  python -m eye_tracking_system_tools.annotation.block_annotator"
