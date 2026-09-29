#!/usr/bin/env bash
# Create/update eye_repo_mac (pip PyQt6 + headless OpenCV; no sudo).
# Usage (from repo root):
#   bash scripts/setup_eye_repo_mac.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Miniforge first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx eye_repo_mac; then
  echo "Updating existing conda env: eye_repo_mac"
  conda env update -f environment_mac.yml --prune
else
  echo "Creating conda env: eye_repo_mac"
  conda env create -f environment_mac.yml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eye_repo_mac

python -c "
import cv2
from PyQt6 import QtCore
import pyqtgraph
import eye_tracking_system_tools
print('OK:', cv2.__version__, QtCore.QT_VERSION_STR)
"

echo ""
echo "Launch examples:"
echo "  conda activate eye_repo_mac"
echo "  python -m eye_tracking_system_tools.annotation.block_annotator"
echo "  python -m eye_tracking_system_tools.annotation.preprocessing_gui"
