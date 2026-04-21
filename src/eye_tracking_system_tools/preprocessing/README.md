# Preprocessing Workflow

This directory contains the preprocessing pipeline for eye-tracking data synchronization, verification, and annotation. Follow the steps below in order to process your raw eye-tracking data.

## Prerequisites

Before starting the preprocessing workflow, ensure that:

1. **Data Structure**: All files are arranged in the proper structure as expected by the `BlockSync` class and `block_generator` function. See the [main README](../README.md) for the required folder structure.

2. **Environment Setup**: The package and all dependencies are installed. See the [main README](../README.md) for installation instructions.

3. **Pupil Annotations**: You have DeepLabCut (or compatible) pupil annotation files in the expected format (one `.csv` file per eye video).

## Optional: Sync-free eye artifacts (parallel track)

**Notebook**: `sync_free_eye_ellipse_pipeline.ipynb`  
**Helpers**: `sync_free_eye_io.py`, `interactive_ellipse_corrector()` in `data_verification_utils.py`

Use this when you want **one row per video frame** (ellipse + Kerr degrees) saved **next to the eye video**, independent of `final_sync_df`, so that regenerating synchronization does not silently desynchronize ellipse CSVs under `analysis/` (see `le_df.csv` / `left_eye_data.csv` vs `final_sync_df.csv` mtimes).

**Invariant:** `eye_frame` in the sync-free tables must use the **same** frame numbering as `L_eye_frame` / `R_eye_frame` in `final_sync_df` (OpenCV frame index convention used by the verification UI).

**Outputs:** CSV + JSON next to each eye `.mp4`; optional join writes `analysis/{left|right}_eye_degrees_from_syncfree_{tag}.csv`.

**Staleness guard:** If `final_sync_df.csv` is newer than those mapped outputs, rerun the mapping step in the notebook.

## Workflow Steps

### Step 1: Block Synchronization

**Notebook**: `block_synchronization.ipynb`

This is the first step in the preprocessing pipeline. This notebook:

- Synchronizes eye-tracking videos with arena videos and electrophysiology recordings
- Creates synchronized dataframes for left and right eye data
- Generates CSV files and artifacts for verification
- Outputs `left_eye_data.csv` and `right_eye_data.csv` in the `analysis/` folder of each block

**What it does:**
- Parses Open Ephys events (or custom synchronization paradigm)
  - Automatically extracts metadata from Open Ephys files (standalone mode, no MATLAB required)
  - Supports both rising and falling edge detection for TTL signals
- Aligns eye video frames to the master arena timebase
- Performs manual correction for alignment accuracy
- Removes camera jitter and LED blink artifacts
- Exports synchronized eye data to CSV files

**Output files:**
- `left_eye_data.csv` - Synchronized left eye tracking data
- `right_eye_data.csv` - Synchronized right eye tracking data
- Various verification artifacts in the `analysis/` folder

---

### Step 2: Data Verification

**Notebook**: `data_verification.ipynb`

After synchronization, use this notebook to verify adherence between eye data and actual eye videos.

**What it does:**
- Loads the synchronized eye data from Step 1
- Provides tools to visualize and verify data alignment
- Allows correction of data orientation issues (horizontal flips, rotations)
- Verifies that eye tracking data matches the actual video frames

**Prerequisites:**
- Must have completed Step 1 (block synchronization)
- Eye data CSV files must exist in the `analysis/` folder

---

### Step 3: Kerr Degree Conversion

**Notebook**: `kerr_degree_conversion.ipynb`

This step calculates gaze vectors from the 2D eye tracking data.

**What it does:**
- Converts 2D pupil center coordinates to 3D gaze vectors
- Calculates gaze angles using the Kerr model
- Processes data for both left and right eyes
- Outputs gaze vector data for downstream analysis

**Prerequisites:**
- Must have completed Steps 1 and 2
- Verified and corrected eye data from previous steps

---

### Step 4: Accelerometer State Annotations

**Notebook**: `add_accelerometer_state_annotations.ipynb`

This is the final step of the core preprocessing pipeline. It adds accelerometer state annotations to the processed eye-tracking data.

**What it does:**
- Integrates accelerometer data with eye-tracking data
- Annotates behavioral states based on accelerometer readings
- Creates final preprocessed datasets ready for analysis

**Prerequisites:**
- Must have completed Steps 1, 2, and 3
- Accelerometer data must be available and properly formatted

---

### Electrophysiology (under development)

**Notebook**: `lfp_led_validation.ipynb`

Validates temporal alignment of LFP extraction to Open Ephys timebase using LED driver events. Use after block synchronization when working with LFP or saccade-triggered analyses.

**Analysis pipelines** (saccade collection, saccade-triggered LFP averages) live in `src/eye_tracking_system_tools/analysis_pipelines/`. Those pipelines and electrophysiology APIs are still under development and may change.

---

## Quick Reference

| Step | Notebook | Purpose | Input | Output |
|------|----------|---------|-------|--------|
| 1 | `block_synchronization.ipynb` | Synchronize videos and recordings | Raw videos, OE files, DLC annotations | `left_eye_data.csv`, `right_eye_data.csv` |
| 2 | `data_verification.ipynb` | Verify data alignment | CSV files from Step 1 | Verified/corrected eye data |
| 3 | `kerr_degree_conversion.ipynb` | Calculate gaze vectors | Verified eye data | Gaze vector data |
| 4 | `add_accelerometer_state_annotations.ipynb` | Add state annotations | Gaze vector data | Final preprocessed data |
| — | `lfp_led_validation.ipynb` | LFP/LED alignment check (electrophysiology, under dev) | Block with OE + LED events | Validation plots |
| — | `sync_free_eye_ellipse_pipeline.ipynb` | DLC → ellipses → verify → Kerr (no sync); optional map to `final_sync_df` | DLC CSV, eye videos, optional `self_kerr_refs.csv` | CSV/JSON beside video; mapped CSV in `analysis/` |

## Additional Resources

- **BlockSync Class**: See `BlockSync_class.py` for the main synchronization class
- **OERecording / Open Ephys unit conversion**: See `OERecording.py`. When `oe_rec.get_data(..., convert_microvolts=True)` is used for neural/headstage channels, the returned values are in **microvolts (µV)**. This matches the Open Ephys `.continuous` format: the header field `bitVolts` is in µV per AD count for headstage channels, so `voltage_µV = raw_int16 * bitVolts`. Per-channel `bitVolts` from each file header is used. For ADC/AUX channels, `get_analog_data` and `get_accel_data` docstrings describe their units (µV and mV respectively).
- **Utility Functions**: See `utility_functions.py` for helper functions including `block_generator`
- **Manual Annotation**: See `manual_outlier_annotation.ipynb` for outlier annotation tools

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the package is installed in development mode: `pip install -e .`

2. **Missing Files**: Verify that your data structure matches the expected format (see main README)

3. **Synchronization Failures**: 
   - Check that Open Ephys events are properly parsed and TTL channels are correctly configured
   - Verify that Open Ephys recording files (`.continuous`, `.events`, `settings.xml`) are present in the `oe_files/` directory
   - The `OERecording` class automatically extracts metadata from these files (standalone mode)

4. **Data Verification Issues**: Ensure that video files are accessible and timestamps are correctly formatted

For more detailed information, refer to the docstrings in each notebook and the main README file.
