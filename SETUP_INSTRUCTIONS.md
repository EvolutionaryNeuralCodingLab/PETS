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

Use **`environment_linux.yml`** (`eye_repo_linux`) first — conda-forge PyQt/OpenCV. Report issues if a dedicated macOS file is needed.

## Environment files

| File | Env name | Platform |
|------|----------|----------|
| `environment_win.yml` | `eye_repo_win` | Windows (pip PyQt6 + opencv-python-headless) |
| `environment_linux.yml` | `eye_repo_linux` | Linux; try on macOS |

Both environments support: Jupyter notebooks, `BlockSync` preprocessing, Block Annotator, Preprocessing GUI, and Event Explorer.

## Jupyter kernel

```bash
conda activate eye_repo_win   # or eye_repo_linux
python -m ipykernel install --user --name eye_repo --display-name "Python (PETS)"
```

In VS Code / Jupyter, select kernel **Python (PETS)**.

## Verify installation

```bash
python examples/smoke_preprocessing_imports.py
```

Headless GUI smoke (optional):

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
pytest tests/test_preprocessing_gui_phase0.py -q
```

## Troubleshooting

- **Wrong Python in Jupyter:** `conda activate eye_repo_win` (or `_linux`) before starting Jupyter.
- **Windows DLL errors with OpenCV + Qt:** use `eye_repo_win` only — do not mix conda `opencv` with pip PyQt6.
- **Linux headless / SSH:** run GUIs with a display, or use `QT_QPA_PLATFORM=offscreen` for pytest only.

## Copying to another machine

Copy the **full repository** (not yml files alone). On the destination PC, run the platform setup script or `conda env create` + `pip install -e .` as above.

Do **not** use `conda env export` snapshots from another machine as the primary env definition.
