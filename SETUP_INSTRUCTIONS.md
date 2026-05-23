# Setup Instructions for eye_repo Environment

## Quick Setup

1. **Activate your existing conda environment:**
   ```bash
   conda activate eye_repo
   ```

2. **Verify Python version:**
   ```bash
   python --version
   # Should show: Python 3.10.19
   ```

3. **Install/update dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Verify pathlib is available:**
   ```bash
   python -c "import pathlib; print('OK')"
   ```

## Jupyter Notebook Setup

**Important**: Make sure your Jupyter notebook is using the `eye_repo` kernel:

1. **Install ipykernel in the eye_repo environment:**
   ```bash
   conda activate eye_repo
   pip install ipykernel
   ```

2. **Register the environment as a Jupyter kernel:**
   ```bash
   python -m ipykernel install --user --name eye_repo --display-name "Python (eye_repo)"
   ```

3. **In Jupyter/VS Code:**
   - Select the kernel: "Python (eye_repo)" or "eye_repo"
   - In VS Code: Press `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose the `eye_repo` environment

## Block Annotator (separate environment)

The annotator uses **PyQt6** and **OpenCV** in one GUI process. On Windows, that conflicts with the **conda `opencv`** stack in `eye_repo`. Use a dedicated environment.

### Windows — copy repo to another PC

Copy the **full PETS repository** (not only a yml file). On the destination machine:

```powershell
cd D:\path\to\PETS
conda env create -f environment_annotator_windows.yml
conda activate eye_annotator
pip install -e . --no-deps
python -m eye_tracking_system_tools.annotation.block_annotator
```

Or: `powershell -ExecutionPolicy Bypass -File scripts\setup_annotator_windows.ps1`

| File | Use |
|------|-----|
| `environment_annotator_windows.yml` | **Portable Windows** — pinned pip deps; then `pip install -e . --no-deps` |
| `environment_annotator.yml` | **Dev on your machine** — minimal; `-e ".[annotator]"` pulls deps from `pyproject.toml` |
| `eye_annotator.yml` | **Do not use** — `conda env export` snapshot; fails on other PCs |

`eye_repo` is unchanged for notebooks and preprocessing; install it with `environment.yml` / `requirements.txt` as above (no PyQt6).

### Ubuntu / Linux without sudo

You do **not** need `apt` or root if conda (Miniforge/Miniconda) is installed in your home directory. Use **`environment_annotator_linux.yml`**, which installs PyQt, OpenCV, and X11/Qt libraries from **conda-forge** instead of system packages.

```bash
cd /path/to/PETS
conda env create -f environment_annotator_linux.yml
conda activate eye_annotator
python -m eye_tracking_system_tools.annotation.block_annotator
```

Or: `bash scripts/setup_annotator_linux.sh`

**What is portable via conda**

| Component | Approach |
|-----------|----------|
| Python, NumPy, SciPy, pandas, h5py, matplotlib | conda-forge |
| PyQt6 GUI | conda-forge `pyqt` (provides `PyQt6` imports) |
| X11 / xcb / EGL / fonts | Pulled in by `pyqt` / `opencv` (e.g. `xcb-util-cursor`, `libgl`) |
| OpenCV (`cv2`) | conda-forge `opencv` on Linux (Windows annotator env uses pip-only to avoid DLL conflicts) |
| `open-ephys-python-tools`, `ellipse` | pip (no conda binary conflict on Linux) |
| This repo | `pip install -e . --no-deps` after conda env create |

**What is not copied from a Windows `conda env export`**

- Windows-only conda packages (`vc`, `ucrt`, `vcomp`, …)
- Windows-only pip packages (`pywinpty`, …)
- A Windows `pip freeze` lock file — Linux needs its own solve (`environment_annotator_linux.yml`), not `requirements-annotator-pinned.txt` from Windows

**What conda cannot install (runtime, not packages)**

- A **display**: the GUI needs `DISPLAY` (SSH X11 forwarding, local desktop, or virtual framebuffer). On a headless node without `DISPLAY`, use conda’s Xvfb and run under it:

  ```bash
  conda activate eye_annotator
  conda install -c conda-forge xorg-xvfb
  xvfb-run -a python -m eye_tracking_system_tools.annotation.block_annotator
  ```

- **GPU drivers** — not required for the annotator (CPU Qt/OpenCV is enough).

## Troubleshooting

### If pathlib import fails:
- Make sure you're using Python 3.4+ (pathlib is standard library)
- Check that you're in the correct conda environment: `conda activate eye_repo`
- Verify the interpreter: `which python` (Linux/Mac) or `where python` (Windows)

### If imports fail:
- Reinstall the package: `pip install -e .`
- Check Python path: `python -c "import sys; print(sys.path)"`
- Verify the package is installed: `pip list | grep eye-tracking`
