# Batch Analysis Scripts

This folder contains scripts and notebooks for **automated, compute-heavy preprocessing steps** that can be run overnight or unattended. These scripts handle the time-consuming synchronization, jitter correction, and DLC processing steps, leaving the quicker verification and analysis steps for interactive work.

## Purpose

The batch analysis scripts are designed to:
- **Run unattended overnight** on large datasets
- **Process multiple blocks** in parallel where possible
- **Save intermediate results** so you can resume or verify later
- **Separate compute-heavy steps** from interactive verification/analysis

After running these batch scripts, you'll have synchronized eye tracking data ready for:
- Interactive verification (choosing Kerr reference points, checking alignment)
- Angle calculations (`calculate_kerr_angles.py`)
- Downstream analyses (saccade detection, LFP alignment, etc.)

---

## Workflow Overview

The typical workflow follows this order:

```
1. batch_block_synchronization.ipynb     [Overnight: ROI selection → brightness → sync]
   ↓
2. batch_jitter_correction.ipynb         [Overnight: Jitter ROI → cross-correlation]
   ↓
3. run_batch_dlc_and_verification.py     [Overnight: DLC → jitter correction → verification GUI]
   ↓
4. Interactive steps                      [Quick: Choose Kerr refs, verify angles, run analyses]
```

**Helper script:**
- `get_animal_block_report.py` - Check which blocks are ready for the next step

---

## File Descriptions

### 1. `batch_block_synchronization.ipynb`

**Purpose:** Synchronize eye tracking videos with Open Ephys TTL events for multiple blocks.

**When to run:** First step in the batch pipeline. Run overnight after setting up ROIs.

**Inputs:**
- Experiment path (contains animal folders)
- Animal name(s) and block numbers
- Channel mapping dictionary (optional; will prompt if missing)
- Manual ROI selection (one-time per block)

**Outputs:**
- `block_path/analysis/<analysis_subfolder_name>/eye_brightness_values_dict.pkl` - Brightness vectors per eye
- `block_path/analysis/<analysis_subfolder_name>/final_sync_df.csv` - Synchronized timestamps
- `block_path/analysis/<analysis_subfolder_name>/eye_left_corrected_sync.csv` - Left eye sync data
- `block_path/analysis/<analysis_subfolder_name>/eye_right_corrected_sync.csv` - Right eye sync data
- `experiment_path/batch_sync_report/sync_log_<analysis_subfolder_name>.txt` - Processing log
- `experiment_path/batch_sync_report/sync_summary_<analysis_subfolder_name>.csv` - Summary table

**Key features:**
- Named analysis subfolder (default: `batch_analysis_output_YYYY_MM_DD`) keeps outputs organized
- Manual ROI selection for brightness computation (one pass through all blocks)
- Parallel brightness computation
- Automatic sync (LED alignment + drift correction) for all blocks
- Self-verification correlation check (L/R brightness at LED blinks)
- Optional: Batch jitter reports (can also run separately via `batch_jitter_correction.ipynb`)

**Usage:**
1. Open notebook and set configuration (experiment path, animal, blocks, analysis subfolder name)
2. Run cells sequentially:
   - Setup blocks and run folders
   - Collect ROIs (manual selection for each block)
   - Compute brightness vectors (parallel)
   - Run synchronization (sequential, writes to log)
   - View summary table
   - (Optional) Run jitter reports

**Time estimate:** ~5-15 minutes per block (depends on video length and number of LED events)

---

### 2. `batch_jitter_correction.ipynb`

**Purpose:** Compute frame-to-frame displacement (jitter) via cross-correlation for multiple blocks.

**When to run:** After `batch_block_synchronization.ipynb` is complete. Can run standalone if sync is already done.

**Inputs:**
- Experiment path, animal(s), block numbers
- Analysis subfolder name (must match the one used in batch sync)
- Jitter ROIs (saved to `jitter_rois.pkl` per block, or selected manually)

**Outputs:**
- `block_path/analysis/<analysis_subfolder_name>/jitter_report_dict.pkl` - Jitter displacement data per eye
- `block_path/analysis/<analysis_subfolder_name>/jitter_rois.pkl` - Saved ROIs (if manually selected)

**Key features:**
- **ROI persistence:** Loads saved ROIs from `jitter_rois.pkl` if present; otherwise prompts for selection and saves them
- **Parallel execution:** One process per block (uses multiple CPU cores)
- **Skip existing:** Blocks with `jitter_report_dict.pkl` are skipped unless `overwrite_jitter_reports=True`
- **Progress bars:** Live progress updates per block (tqdm)

**Usage:**
1. Open notebook and set configuration (must match analysis subfolder from batch sync)
2. Run cells:
   - Setup blocks
   - ROI selection (loads from disk if saved, otherwise prompts)
   - Jitter computation (parallel, skips existing reports)

**Time estimate:** ~10-30 minutes per block (depends on video length and ROI size)

---

### 3. `run_batch_dlc_and_verification.py`

**Purpose:** Run DLC processing, jitter correction, LED cleanup, and interactive verification GUI for blocks that were successfully synced.

**When to run:** After both `batch_block_synchronization.ipynb` and `batch_jitter_correction.ipynb` are complete.

**Inputs:**
- Path to sync log file from `batch_block_synchronization.ipynb`:
  - `experiment_path/batch_sync_report/sync_log_<analysis_subfolder_name>.txt`
- Channel mapping JSON (optional; will try to load from block's `ttl_manual_mapping.json`)

**Outputs:**
- `block_path/analysis/<analysis_subfolder_name>/dlc_data/` - DLC tracking results
- `block_path/analysis/<analysis_subfolder_name>/left_eye_data.csv` - Final left-eye tracking data (2D)
- `block_path/analysis/<analysis_subfolder_name>/right_eye_data.csv` - Final right-eye tracking data (2D)
- Corrected eye data (after interactive verification GUI)

**Key features:**
- **Reads sync log:** Automatically extracts experiment path, animal, analysis subfolder, and synced block numbers
- **DLC pipeline:** Initialization → DLC → jitter correction → LED blink removal → `find_jittery_frames` → `create_eye_data`
- **Jitter safety check:** Flags blocks where >10% of frames are removed by `find_jittery_frames` (prompts to inspect)
- **Interactive verification:** GUI for each block/eye to verify and correct eye tracking data
- **Progress bar:** Terminal progress bar for verification step

**Usage:**
```bash
python -m eye_tracking_system_tools.batch_analysis.run_batch_dlc_and_verification \
    path/to/sync_log_batch_analysis_output_YYYY_MM_DD.txt

# Skip verification GUI (DLC only):
python -m eye_tracking_system_tools.batch_analysis.run_batch_dlc_and_verification \
    path/to/sync_log_batch_analysis_output_YYYY_MM_DD.txt --no-verify

# Provide channel mapping JSON:
python -m eye_tracking_system_tools.batch_analysis.run_batch_dlc_and_verification \
    path/to/sync_log_batch_analysis_output_YYYY_MM_DD.txt --channeldict-json path/to/channeldict.json

# Force processing even if jitter removal is high:
python -m eye_tracking_system_tools.batch_analysis.run_batch_dlc_and_verification \
    path/to/sync_log_batch_analysis_output_YYYY_MM_DD.txt --skip-jittery-check
```

**Time estimate:** 
- DLC processing: ~30-60 minutes per block (depends on video length)
- Verification GUI: ~2-5 minutes per video (interactive)

---

### 4. `get_animal_block_report.py`

**Purpose:** Generate a report listing all blocks for given animals, their analysis folders, and which blocks are "ready" (have both brightness and jitter pickles).

**When to run:** Anytime to check pipeline status or find which blocks are ready for the next step.

**Inputs:**
- Experiment path
- Animal name(s)

**Outputs:**
- CSV report with columns: `animal`, `experiment_date`, `block_num`, `block_path`, `analysis_folder`, `analysis_path`, `files_in_analysis`, `has_brightness_pickle`, `has_jitter_pickle`, `ready`
- Console summary of ready blocks

**Usage:**
```bash
# Single animal:
python -m eye_tracking_system_tools.batch_analysis.get_animal_block_report \
    /path/to/experiment PV_126

# Multiple animals:
python -m eye_tracking_system_tools.batch_analysis.get_animal_block_report \
    /path/to/experiment PV_126 PV_106

# Save to CSV:
python -m eye_tracking_system_tools.batch_analysis.get_animal_block_report \
    /path/to/experiment PV_126 -o report.csv
```

**Example output:**
```
BLOCKS READY FOR FURTHER PROCESSING
(have both eye_brightness pickle and jitter_report_dict.pkl)
============================================================
  PV_126  date=2025_02_04  block=6 / batch_analysis_output_2025_02_05
    -> D:\data\PV_126\2025_02_04\block_006\analysis\batch_analysis_output_2025_02_05
  PV_126  date=2025_02_04  block=7 / batch_analysis_output_2025_02_05
    -> D:\data\PV_126\2025_02_04\block_007\analysis\batch_analysis_output_2025_02_05
```

---

### 5. `promote_latest_analysis_outputs.py` and `promote_latest_analysis_outputs.ipynb`

**Purpose:** Promote the newest heavy-computation files from dated analysis subfolders (for example `analysis/batch_analysis_output_2026_02_05/`) and the analysis root itself to the analysis root (`analysis/`), so default `BlockSync` initialization automatically loads the latest data.

**When to run:** After running one or more batch pipelines across different dates/names and before interactive per-block work.

**Default promoted files:**
- Brightness: newest of `eye_brightness_values_dict.pkl` or `eye_brightness.pickle` -> `analysis/eye_brightness_values_dict.pkl`
- Jitter: newest `jitter_report_dict.pkl` -> `analysis/jitter_report_dict.pkl`
- Sync: newest `final_sync_df.csv` -> `analysis/final_sync_df.csv`

**Behavior:** The script considers both the analysis root and its subfolders when choosing the "most recent" file by mtime. If the current top-level file is already the newest, it is left in place (no copy).

**Notebook:** Use `promote_latest_analysis_outputs.ipynb` to set parameters (experiment path, animals, prefix, dry_run, etc.) in one cell and run the promotion in the next.

**Safety behavior:**
- Existing root files are backed up by default to `analysis/__promote_backup__/<timestamp>/`
- `--dry-run` mode previews all changes without copying files

**Usage:**
```bash
# Preview changes only:
python -m eye_tracking_system_tools.batch_analysis.promote_latest_analysis_outputs \
    /path/to/experiment PV_126 PV_106 --dry-run

# Apply promotions (default source prefix: batch_analysis_output_):
python -m eye_tracking_system_tools.batch_analysis.promote_latest_analysis_outputs \
    /path/to/experiment PV_126 PV_106

# Include every analysis subfolder as source (not just batch_analysis_output_*):
python -m eye_tracking_system_tools.batch_analysis.promote_latest_analysis_outputs \
    /path/to/experiment PV_126 --source-prefix ""

# Save a detailed report:
python -m eye_tracking_system_tools.batch_analysis.promote_latest_analysis_outputs \
    /path/to/experiment PV_126 -o promote_report.csv
```

---

## Folder Structure

After running the batch scripts, your data structure will look like:

```
experiment_path/
├── batch_sync_report/                          # Report folder (created by batch sync)
│   ├── sync_log_batch_analysis_output_YYYY_MM_DD.txt
│   └── sync_summary_batch_analysis_output_YYYY_MM_DD.csv
│
└── animal_name/
    └── date_folder/
        └── block_XXX/
            └── analysis/
                └── batch_analysis_output_YYYY_MM_DD/    # Named analysis subfolder
                    ├── eye_brightness_values_dict.pkl
                    ├── final_sync_df.csv
                    ├── eye_left_corrected_sync.csv
                    ├── eye_right_corrected_sync.csv
                    ├── jitter_report_dict.pkl
                    ├── jitter_rois.pkl
                    ├── dlc_data/
                    ├── left_eye_data.csv
                    ├── right_eye_data.csv
                    └── ... (other outputs)
```

---

## Tips for Overnight Runs

1. **Use consistent analysis subfolder names:** Set `analysis_subfolder_name` explicitly (e.g., `"batch_run_20250205"`) so you can reference it later.

2. **Check readiness before starting:** Run `get_animal_block_report.py` to see which blocks already have brightness/jitter pickles (can skip those).

3. **Monitor progress:** 
   - Batch sync writes to `batch_sync_report/sync_log_*.txt` (check this if a run fails)
   - Jitter computation shows progress bars per block
   - DLC script prints status per block

4. **Resume interrupted runs:**
   - Batch sync: Set `previous_analysis_subfolder_name` to continue from a previous run
   - Jitter: Already-computed reports are skipped automatically
   - DLC: Re-run the script if needed; final exports are `left_eye_data.csv` and `right_eye_data.csv` (use `--overwrite-dlc` to force re-read of DLC data)

5. **Handle failures:**
   - Check the sync log for error messages
   - Failed blocks are marked in the summary CSV
   - Process failed blocks individually using `block_synchronization.ipynb` or `block_synchronization.ipynb` in the `preprocessing/` folder

6. **Multi-animal support:**
   - `batch_jitter_correction.ipynb` supports multiple animals: `animal = ["PV_126", "PV_106"]`, `block_numbers = [[6], [15]]`
   - `batch_block_synchronization.ipynb` processes one animal at a time (run multiple times for multiple animals)

---

## Next Steps After Batch Processing

Once batch processing is complete:

1. **Verify synchronization:** Check the sync summary CSV and sync log for any low-correlation or failed blocks.

2. **Interactive verification:** If you ran `run_batch_dlc_and_verification.py` with verification, you've already done this. Otherwise, use `data_verification.ipynb` in `preprocessing/`.

3. **Choose Kerr reference points:** Use `kerr_degree_conversion.ipynb` or `calculate_kerr_angles.py` to set reference points and compute angles.

4. **Run downstream analyses:** Use notebooks in `analysis_pipelines/` (e.g., `saccade_collection_pipeline.ipynb`, `saccade_lfp_average_pipeline.ipynb`).

---

## Dependencies

All scripts import from:
- `eye_tracking_system_tools.preprocessing` - Core preprocessing functions
- `eye_tracking_system_tools.preprocessing.BlockSync_class` - BlockSync class
- `eye_tracking_system_tools.preprocessing.block_sync_core` - Sync and jitter functions
- `eye_tracking_system_tools.preprocessing.data_verification_utils` - Verification GUI

Make sure these modules are in your Python path (install the package or add `src/` to `PYTHONPATH`).
