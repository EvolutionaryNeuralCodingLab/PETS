# Create/update eye_annotator on Windows and install this repo in editable mode.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_annotator_windows.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "conda not found. Install Miniconda/Anaconda first."
}

$envExists = conda env list | Select-String -Pattern "^\s*eye_annotator\s"
if ($envExists) {
    Write-Host "Updating existing conda env: eye_annotator"
    conda env update -f environment_annotator_windows.yml --prune
} else {
    Write-Host "Creating conda env: eye_annotator"
    conda env create -f environment_annotator_windows.yml
}

conda activate eye_annotator
if ($LASTEXITCODE -ne 0) {
    Write-Host "If 'conda activate' failed in this script, run manually:"
    Write-Host "  conda activate eye_annotator"
    Write-Host "  pip install -e . --no-deps"
    exit 1
}

pip install -e . --no-deps

Write-Host "Verifying imports..."
python -c @"
import cv2
from PyQt6 import QtCore
import pyqtgraph
import open_ephys.analysis  # noqa: F401
import eye_tracking_system_tools
print('OK:', cv2.__version__, QtCore.QT_VERSION_STR)
"@

Write-Host ""
Write-Host "Launch:"
Write-Host "  conda activate eye_annotator"
Write-Host "  python -m eye_tracking_system_tools.annotation.block_annotator"
