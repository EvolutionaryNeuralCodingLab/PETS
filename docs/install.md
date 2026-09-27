# Install

Python 3.10 or higher, with Conda (recommended) or pip.

| Platform | File | Env name |
|----------|------|----------|
| Windows | `environment_win.yml` | `eye_repo_win` |
| Linux | `environment_linux.yml` | `eye_repo_linux` |
| macOS | `environment_mac.yml` | `eye_repo_mac` |

**Windows**

```powershell
conda env create -f environment_win.yml
conda activate eye_repo_win
pip install -e .
```

Or: `powershell -ExecutionPolicy Bypass -File scripts\setup_eye_repo_windows.ps1`

**Linux**

```bash
conda env create -f environment_linux.yml
conda activate eye_repo_linux
```

(`environment_linux.yml` installs the package editable via pip.)

Or: `bash scripts/setup_eye_repo_linux.sh`

**macOS**

```bash
conda env create -f environment_mac.yml
conda activate eye_repo_mac
```

Or: `bash scripts/setup_eye_repo_mac.sh`

GUI dependencies use pip PyQt6 + headless OpenCV on Windows and macOS. Linux uses conda-forge PyQt and OpenCV.

## Check the install

```bash
python examples/smoke_preprocessing_imports.py
```

Then the three user paths:

- Reproduce paper plots: [figures/README.md](../figures/README.md)
- Preprocess a block: [preprocessing.md](preprocessing.md)
- Raspberry Pi acquisition: [acquisition.md](acquisition.md)
