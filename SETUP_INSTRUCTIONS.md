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

## Troubleshooting

### If pathlib import fails:
- Make sure you're using Python 3.4+ (pathlib is standard library)
- Check that you're in the correct conda environment: `conda activate eye_repo`
- Verify the interpreter: `which python` (Linux/Mac) or `where python` (Windows)

### If imports fail:
- Reinstall the package: `pip install -e .`
- Check Python path: `python -c "import sys; print(sys.path)"`
- Verify the package is installed: `pip list | grep eye-tracking`
