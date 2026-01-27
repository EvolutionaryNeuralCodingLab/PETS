# Preprocessing Workflow

This directory contains the preprocessing pipeline for eye-tracking data synchronization, verification, and annotation. Follow the steps below in order to process your raw eye-tracking data.

## Prerequisites

Before starting the preprocessing workflow, ensure that:

1. **Data Structure**: All files are arranged in the proper structure as expected by the `BlockSync` class and `block_generator` function. See the [main README](../README.md) for the required folder structure.

2. **Environment Setup**: The package and all dependencies are installed. See the [main README](../README.md) for installation instructions.

3. **Pupil Annotations**: You have DeepLabCut (or compatible) pupil annotation files in the expected format (one `.csv` file per eye video).

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

This is the final step of the preprocessing pipeline. It adds accelerometer state annotations to the processed eye-tracking data.

**What it does:**
- Integrates accelerometer data with eye-tracking data
- Annotates behavioral states based on accelerometer readings
- Creates final preprocessed datasets ready for analysis

**Prerequisites:**
- Must have completed Steps 1, 2, and 3
- Accelerometer data must be available and properly formatted

---

## Quick Reference

| Step | Notebook | Purpose | Input | Output |
|------|----------|---------|-------|--------|
| 1 | `block_synchronization.ipynb` | Synchronize videos and recordings | Raw videos, OE files, DLC annotations | `left_eye_data.csv`, `right_eye_data.csv` |
| 2 | `data_verification.ipynb` | Verify data alignment | CSV files from Step 1 | Verified/corrected eye data |
| 3 | `kerr_degree_conversion.ipynb` | Calculate gaze vectors | Verified eye data | Gaze vector data |
| 4 | `add_accelerometer_state_annotations.ipynb` | Add state annotations | Gaze vector data | Final preprocessed data |

## Additional Resources

- **BlockSync Class**: See `BlockSync_class.py` for the main synchronization class
- **Utility Functions**: See `utility_functions.py` for helper functions including `block_generator`
- **Manual Annotation**: See `manual_outlier_annotation.ipynb` for outlier annotation tools

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the package is installed in development mode: `pip install -e .`

2. **Missing Files**: Verify that your data structure matches the expected format (see main README)

3. **Synchronization Failures**: Check that Open Ephys events are properly parsed and TTL channels are correctly configured

4. **Data Verification Issues**: Ensure that video files are accessible and timestamps are correctly formatted

For more detailed information, refer to the docstrings in each notebook and the main README file.
