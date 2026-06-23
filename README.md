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

### Installation (recommended)

One conda environment **`eye_repo`** covers notebooks, preprocessing, the Preprocessing GUI,
Block Annotator, and Event Explorer. GUI dependencies use **pip** PyQt6 and headless OpenCV
to avoid Windows DLL conflicts.

```bash
conda env create -f environment.yml
conda activate eye_repo
pip install -e .
```

**Linux:** prefer `environment_linux.yml` (conda-forge PyQt + OpenCV).

**Legacy:** `environment_annotator*.yml` and `eye_annotator` are deprecated; use `eye_repo`.

Alternative installs (`pip install -e .` only, or `requirements.txt`) still work if you manage
compatible versions yourself.

### Verification

Run the smoke test to verify imports:

```bash
python examples/smoke_preprocessing_imports.py
```

## Block Annotator GUI

Standalone PyQt6 app for synchronized review of arena + eye videos, Open Ephys traces, and event annotations on blocks with `analysis/final_sync_df.csv`.

### Install and launch

Use the unified **`eye_repo`** environment (see Setup above).

```powershell
conda activate eye_repo
python -m eye_tracking_system_tools.annotation.block_annotator --block "D:\path\to\block_015" --output "D:\path\to\annotator_output"
```

Add `--dialog` to show the setup dialog. Optional env vars: `PETS_ANNOTATOR_BLOCK`, `PETS_ANNOTATOR_OUTPUT`.

Headless check (reads frames + builds QPixmap):

```powershell
python scripts/validate_block_annotator_load.py --block "D:\path\to\block_015" --output "D:\path\to\annotator_output"
```

On first run (dialog mode), pick an **output folder** (required). If `annotator_config.yaml` is missing there, a template is created with default event types: `saccade`, `blink`, `noise`, `pupil event`. Then select a **block folder** containing `analysis/final_sync_df.csv`.

### Config (`annotator_config.yaml`)

```yaml
event_types:
  - saccade
  - blink
  - noise
  - pupil event
default_range_half_width_ms: 100.0
playback_fps: 60.0
step_rows: 1
```

### Per-block output JSON

Saved under the output folder as `{animal}_{date}_block_{num}_annotations.json` (date segment omitted when unknown). Schema version 1; each event has `timepoint_ms`, `start_ms`, `end_ms`, `range_half_width_ms`, frame IDs, and optional `note`. **Save replaces the full event list** (no merge-by-id in v1).

### Controls

- Transport: play/pause, slider, step buttons; speed 0.25×–4× (slow motion advances every row; fast-forward may skip rows).
- Keyboard: Space (play/pause), R (reverse play), `[` / `]` (step ±1 row), `{` / `}` (jump ±N rows), M (mark event). See **Help → Keyboard shortcuts** in the app.
- Arena: dropdown selects one arena MP4; L/R panels support raw display flip (display only) or annotated ellipse overlay.
- OE trace: stream dropdown (HS / ADC / AUX), downsample factor for overview plot, playhead synced to `ms_axis`.

### Tests

```bash
pytest tests/test_block_annotator.py -q
```

## Event Explorer GUI

Second-stage PyQt6 app for browsing Block Annotator `*_annotations.json` events, inspecting time-aligned eye and EP traces (±ms window around `timepoint_ms = 0`), multi-trial averages, and NPZ export. **No video playback in v1.**

### Install and launch

Same **`eye_repo`** env as the Block Annotator:

```powershell
conda activate eye_repo
python -m eye_tracking_system_tools.annotation.event_explorer --help
```

```powershell
python -m eye_tracking_system_tools.annotation.event_explorer --json _annotator_out\PV_106_2025_09_04_block_015_annotations.json
```

Optional CLI: `--json PATH` (repeatable), `--json-list PATH`, `--scan-dir ROOT` (repeatable). With no args, the app opens and prompts **Add sources…** (JSON files, folders, recursive scan).

### Session file (`*.explorer_session.json`)

Manual **File → Save session**. Stores catalog sources, block path remap table, visible columns, ±window ms, stream toggles, OE HS channel list, normalization mode, and optional column overrides (`pupil_column`, `l_degrees_column`, `r_degrees_column`). On exit, if the session changed, you are prompted to save.

### Export

**File → Export selection** writes `explorer_export_{timestamp}.npz` and a sidecar `.json` (provenance, alignment, load-log excerpt) to a folder you choose.

### Load log

The **Load log** dock and an auto-written `explorer_load_{timestamp}.log` record eye CSV candidate lists, newest-file choice, frame vs `ms_axis` fallback, stale-sync decisions, and remap actions.

### Tests

```bash
pytest tests/test_event_explorer.py -q
```

## Preprocessing GUI

Tabbed PyQt6 app that runs the five preprocessing notebooks (sync, verify, Kerr, behavior,
optional sync-free) on a `block_xxx/` folder. Replaces interactive notebook cells with native
widgets (pyqtgraph plots, ellipse verifier, manual TTL dialog).

### Launch

```powershell
conda activate eye_repo
python -m eye_tracking_system_tools.annotation.preprocessing_gui `
  --experiment-path D:\path\to\experiment --animal PV_106 --block 015
```

Add `--dialog` to pick experiment/animal/blocks interactively. Config persists as
`preproc_gui_config.yaml` in your chosen output folder.

### Tutorial — full block preprocessing for downstream analysis

**Prerequisites:** a `block_xxx/` folder with arena + eye videos, `oe_files/` (or
`parsed_events.csv`), and DeepLabCut CSVs in each eye folder (`eye_videos/LE/...`,
`eye_videos/RE/...`). See [Required Data Structure](#required-data-structure) below.

| Step | Tab | Action | You get |
|------|-----|--------|---------|
| 1 | **Sync** | Prepare data → parse OE events → extract brightness → build arena grid → simple sync → (optional Bokeh shift plot) apply shifts → build final sync → verify → export | `final_sync_df.csv` |
| 2 | **Sync** | Read DLC + fit ellipses → jitter report → correct jitter & LED blinks → preview/remove outliers → finalize eye data | `left_eye_data.csv`, `right_eye_data.csv` |
| 3 | **Verify** | Review ellipses on both videos, click Kerr refs, **Save & export (both eyes)** | `self_kerr_refs.csv`, corrected eye CSVs |
| 4 | **Kerr** | Set `name_tag` (e.g. `raw_verified`) → Calculate → Export merged | `left/right_kerr_angle_<tag>.csv`, `left/right_eye_data_<tag>.csv` |
| 5 | **Behavior** *(optional)* | Requires `lizMov.mat` from MATLAB `getLizMovement` | `block_<NNN>_behavior_state.csv` |
| 6 | **Sync-free** *(optional alternate path)* | Run ellipses → verify → **Finalize** | Per eye: `*_eye_data.csv`, `*_kerr_refs.csv`, `*_meta.json`; optional `analysis/*_timeline.csv` |

After step 4 you typically have everything needed for **Block Annotator** and **Event Explorer**
(`final_sync_df.csv` + Kerr-annotated eye data). Behavior adds movement state segments.
Sync-free is an alternate ellipse/Kerr path tied to raw eye videos.

**Reference sample blocks:**

- Sync / verify / Kerr: `PV_106 / 2025_09_04 / block_015`
- Behavior (`lizMov.mat`): `PV_126 / 2024_07_18 / block_006`

### Tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
pytest tests/test_preprocessing_gui_*.py -q
python scripts/validate_preprocessing_gui_load.py `
  --experiment-path D:\sample_data_for_eye_repo --animal PV_106 --block 015
```

See `development/PREPROCESSING_GUI_ACCEPTANCE.md` for scope and known gaps.

## Project Structure

```
src/eye_tracking_system_tools/
├── annotation/          # Block Annotator + Event Explorer GUIs
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


