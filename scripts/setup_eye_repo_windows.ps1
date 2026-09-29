# Create/update eye_repo_win on Windows and install this repo in editable mode.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_eye_repo_windows.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "conda not found. Install Miniconda/Anaconda first."
}

$envExists = conda env list | Select-String -Pattern "^\s*eye_repo_win\s"
if ($envExists) {
    Write-Host "Updating existing conda env: eye_repo_win"
    conda env update -f environment_win.yml --prune
} else {
    Write-Host "Creating conda env: eye_repo_win"
    conda env create -f environment_win.yml
}

conda activate eye_repo_win
if ($LASTEXITCODE -ne 0) {
    Write-Host "If 'conda activate' failed in this script, run manually:"
    Write-Host "  conda activate eye_repo_win"
    Write-Host "  pip install -e ."
    exit 1
}

pip install -e .

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
Write-Host "Launch examples:"
Write-Host "  conda activate eye_repo_win"
Write-Host "  python -m eye_tracking_system_tools.annotation.block_annotator"
Write-Host "  python -m eye_tracking_system_tools.annotation.preprocessing_gui"
