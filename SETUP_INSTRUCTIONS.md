# Setup Instructions for PETS

## Quick setup

### Windows

```powershell
cd D:\path\to\PETS
conda env create -f environment_win.yml
conda activate eye_repo_win
pip install -e .
```

Or: `powershell -ExecutionPolicy Bypass -File scripts\setup_eye_repo_windows.ps1`

### Linux

```bash
cd /path/to/PETS
conda env create -f environment_linux.yml
conda activate eye_repo_linux
```

Or: `bash scripts/setup_eye_repo_linux.sh`

### macOS

```bash
cd /path/to/PETS
conda env create -f environment_mac.yml
conda activate eye_repo_mac
```

Or: `bash scripts/setup_eye_repo_mac.sh`

Do **not** use `environment_linux.yml` on macOS — it pulls Linux-only packages (`libegl`, `xcb-util-cursor`) that are unavailable on `osx-arm64` / `osx-64`.

## Environment files

| File | Env name | Platform |
|------|----------|----------|
| `environment_win.yml` | `eye_repo_win` | Windows (pip PyQt6 + opencv-python-headless) |
| `environment_linux.yml` | `eye_repo_linux` | Linux (conda-forge PyQt + OpenCV) |
| `environment_mac.yml` | `eye_repo_mac` | macOS (pip PyQt6 + opencv-python-headless) |

Both environments support: Jupyter notebooks, `BlockSync` preprocessing, Block Annotator, Preprocessing GUI, and Event Explorer.

## Jupyter kernel

```bash
conda activate eye_repo_win   # or eye_repo_linux / eye_repo_mac
python -m ipykernel install --user --name eye_repo --display-name "Python (PETS)"
```

In VS Code / Jupyter, select kernel **Python (PETS)**.

## Verify installation

```bash
python examples/smoke_preprocessing_imports.py
```

Headless GUI smoke (optional):

```bash
export QT_QPA_PLATFORM=offscreen   # macOS / Linux
pytest tests/test_preprocessing_gui_phase0.py -q
```

Windows PowerShell: `$env:QT_QPA_PLATFORM = "offscreen"` before `pytest`.

## Troubleshooting

- **Wrong Python in Jupyter:** `conda activate eye_repo_win` (or `_linux`) before starting Jupyter.
- **Windows DLL errors with OpenCV + Qt:** use `eye_repo_win` only — do not mix conda `opencv` with pip PyQt6.
- **Linux headless / SSH:** run GUIs with a display, or use `QT_QPA_PLATFORM=offscreen` for pytest only.

## Copying to another machine

Copy the **full repository** (not yml files alone). On the destination PC, run the platform setup script or `conda env create` + `pip install -e .` as above.

Do **not** use `conda env export` snapshots from another machine as the primary env definition.
