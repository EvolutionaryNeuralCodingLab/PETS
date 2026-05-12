# PETS (Personalized Eye Tracking System)

This repository supplements research work under review.

## Overview

This repository contains the 3D models and Fusion360 environment we use to define camera-eye geometry and create printable personalized eye-tracking headstages for various animals.  
It also holds the associated Raspberry Pi code for eye cameras, and provides tools for synchronizing eye-tracking videos with arena videos and electrophysiology recordings, preprocessing eye-tracking data, and reproducing figures from our research. 
The synchronization pipeline is designed to work with various recording formats, with Open Ephys provided as a reference implementation.

## Setup

### Prerequisites

- Python 3.10 or higher
- Conda (recommended) or pip

### Installation Options
You have three options for setting up the environment:

#### Option 1: Using Conda Environment (recommanded)

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate eye_repo

# Install package in development mode
pip install -e .
```

#### Option 2: Using pyproject.toml

```bash
pip install -e .
```

This will install the package and all dependencies in development mode.

#### Option 3: Using requirements.txt

```bash
pip install -r requirements.txt
pip install -e .
```

### Verification

Run the smoke test to verify imports:

```bash
python examples/smoke_preprocessing_imports.py
```

## Project Structure

```
src/eye_tracking_system_tools/
├── preprocessing/       # Data synchronization and preprocessing
│   ├── block_synchronization.ipynb
│   ├── data_verification.ipynb
│   ├── kerr_degree_conversion.ipynb
│   ├── add_accelerometer_state_annotations.ipynb
│   ├── lfp_led_validation.ipynb   # LED-based LFP alignment verification
│   ├── BlockSync_class.py
│   ├── OERecording.py             # Open Ephys continuous data access (standalone)
│   └── ...
├── analysis_pipelines/   # Downstream analysis (electrophysiology under development)
│   ├── saccade_collection_pipeline.ipynb
│   └── saccade_lfp_average_pipeline.ipynb
├── figures/             # Figure reproduction scripts
├── raspberry_pi/        # Raspberry Pi video capture utilities
└── utils/               # General utilities
    ├── 3D_printing_files/  # 3D printing files and virtual fitting guide
    └── meshroom_pipeline_template.mg
```

## Electrophysiology (under development)

Open Ephys integration and LFP-related workflows are **still under development**. The following are available on feature branches but APIs and pipelines may change:

- **Standalone Open Ephys access**: `OERecording` in `preprocessing/OERecording.py` reads `.continuous` and events without MATLAB; `get_data(..., convert_microvolts=True)` returns data in µV.
- **LED-based LFP verification**: `preprocessing/lfp_led_validation.ipynb` checks temporal alignment of LFP to LED events.
- **Analysis pipelines**: `analysis_pipelines/saccade_collection_pipeline.ipynb` and `saccade_lfp_average_pipeline.ipynb` for saccade collection and saccade-triggered LFP averages.

See `src/eye_tracking_system_tools/preprocessing/README.md` for preprocessing details and `src/eye_tracking_system_tools/analysis_pipelines/` for pipeline notebooks.

## Usage

### Figure Reproduction

The figure reproduction scripts are straightforward to use. Each script in `src/eye_tracking_system_tools/figures/reproduction/main_figures/` can be run directly to reproduce the corresponding paper figure.

**Example:**
```bash
cd src/eye_tracking_system_tools/figures/reproduction/main_figures/Fig_1_e
python figure_1e.py
```

### Data Preprocessing and Synchronization

The preprocessing module requires data organized in a specific folder structure that matches the `BlockSync` class expectations.

#### Required Data Structure

The `BlockSync` class expects the following folder structure:

```
path_to_animal_folder/
└── animal_call/
    └── experiment_date/  (format: yyyy_mm_dd, or None for no date paradigm)
        └── block_xxx/
            ├── arena_videos/          # External arena video outputs
            ├── eye_videos/
            │   ├── LE/                 # Left eye videos
            │   │   └── video_folder/
            │   │       ├── video.h264  # Video file
            │   │       ├── video.mp4   # Video file (optional)
            │   │       ├── DLC_analysis_file.csv  # DeepLabCut pupil annotations
            │   │       └── timestamps.csv         # Video timestamps
            │   └── RE/                 # Right eye videos
            │       └── video_folder/
            │           ├── video.h264
            │           ├── video.mp4
            │           ├── DLC_analysis_file.csv
            │           └── timestamps.csv
            ├── oe_files/               # Open Ephys recordings (or custom format)
            │   └── experiment_datetime/  # e.g., "PV106_IMU_trial4_prey_2025-09-04_13-24-17"
            │       ├── Record Node XXX/   # e.g., "Record Node 106" (contains actual recording files)
            │       │   ├── settings.xml    # Recording settings
            │       │   ├── structure.openephys  # Structure file (Open Ephys 0.6+)
            │       │   ├── *.continuous    # Continuous channel files (e.g., 104_RhythmData_CH1.continuous)
            │       │   ├── *.events        # Event files (e.g., all_channels.events, 104_RhythmData.events)
            │       │   └── *.timestamps    # Timestamp files (if present)
            │       ├── events.csv         # Event data (auto-generated by BlockSync from .events file)
            │       └── parsed_events.csv   # Parsed TTL events (auto-generated by BlockSync)
            └── analysis/               # Output directory (initially empty)
```

#### Pupil Annotations

**Important:** This repository does not provide the pupil annotation model. Users must provide their own pupil annotations that adhere to the DeepLabCut `.csv` export format, with one annotation file per eye video.

If you use a different pupil annotation method, you can work around this requirement by converting your annotations to match the DeepLabCut format, or by modifying the preprocessing code to accept your format.

#### Synchronization Pipeline

**Basic Usage:**

```python
from eye_tracking_system_tools.preprocessing import BlockSync

# Initialize block synchronization
block = BlockSync(
    animal_call="animal_name",
    experiment_date="yyyy_mm_dd",  # or None for no date paradigm
    block_num="001",
    path_to_animal_folder="/path/to/data",
    channeldict=None  # Optional: custom channel mapping
)

# Parse synchronization events
block.parse_open_ephys_events()

# Run synchronization
block.synchronize_block()
```

**Open Ephys Integration:**

The `OERecording` class can work in two modes:
1. **Standalone mode (default)**: Automatically extracts metadata directly from Open Ephys recording files (`.continuous`, `.events`, `settings.xml`). No external dependencies required.
2. **Legacy mode**: Can load metadata from MATLAB-generated `.mat` files for backward compatibility with existing workflows.

The standalone mode eliminates the need for MATLAB preprocessing, making the pipeline fully self-contained within this repository.

**Custom Synchronization Paradigms:**

The synchronization pipeline is not limited to Open Ephys recording formats. While Open Ephys is provided as a reference implementation, users can parse their own synchronization paradigm by creating a `parsed_events.csv` file that matches the expected format.

The `parsed_events.csv` file should be a pandas DataFrame (saved as CSV) with the following structure:

- **Timestamp columns:** One column per synchronization channel containing timestamps (e.g., `Arena_TTL`, `L_eye_TTL`, `R_eye_TTL`)
- **Frame columns:** Corresponding frame number columns with `_frame` suffix (e.g., `Arena_TTL_frame`, `L_eye_TTL_frame`, `R_eye_TTL_frame`)

To use a custom synchronization paradigm:

1. Create your `parsed_events.csv` file in the `oe_files/experiment_datetime/` directory
2. Ensure it follows the format described above
3. The `BlockSync` class will automatically detect and use this file if it exists

For detailed examples and usage, see `src/eye_tracking_system_tools/preprocessing/block_synchronization.ipynb`.


