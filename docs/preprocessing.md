# Preprocessing GUI

The Preprocessing GUI is the way to process an acquired block. It is a tabbed PyQt6 app on a `block_xxx/` folder.

```bash
conda activate eye_repo_mac   # or eye_repo_win / eye_repo_linux
python -m eye_tracking_system_tools.annotation.preprocessing_gui
```

Add `--dialog` to pick experiment, animal, and blocks interactively. To open one block without that dialog, pass the folder that contains the animal folder, the animal, the block, and an output folder:

```bash
python -m eye_tracking_system_tools.annotation.preprocessing_gui \
  --experiment-path /path/to/folder_that_contains_PV_106 \
  --animal PV_106 \
  --block 015 \
  --output /path/to/gui_output
```

`--experiment-path` is the parent of `PV_106`. `--output` is required to skip the dialog. Config is saved as `preproc_gui_config.yaml` in that output folder.

Worked example (shortest analyzed block): **`PV_106 / 2025_09_04 / block_015`**. It is not stored in this software repository. Download `PV_106.zip` from the associated dataset ([record](https://zenodo.org/records/23016364), DOI [10.5281/zenodo.23016364](https://doi.org/10.5281/zenodo.23016364)) and unzip it. That zip contains the example block, including the videos and Open Ephys files. The example already includes `lizMov.mat`, so the Behavior tab can load it. It does not include an accelerometer calibration file, so **Compute lizMov (Python)** uses built-in defaults and says so in the terminal. If the last Open Ephys record is a few samples short, that command prints whether only record padding was missing or whether the end of the accelerometer window was left as zeros, and what that means for eye/arena synchronization versus head-movement timing.

## Recommended tab order

| Step | Tab | Result |
|------|-----|--------|
| 1 | **Sync** | `analysis/final_sync_df.csv`, then `left_eye_data.csv` / `right_eye_data.csv` |
| 2 | **Verify** | overlays, Kerr refs, corrected eye CSVs |
| 3 | **Kerr** | `left/right_kerr_angle_<tag>.csv` and merged eye CSVs |
| 4 | **Calibration** | `analysis/LR_pix_size.csv` (needed for pupil / jitter) |
| 5 | **Behavior** (optional) | `block_<NNN>_behavior_state.csv` from accelerometer / `lizMov.mat` |
| 6 | **Saccades** | `analysis/saccades/` (`saccade_events.csv` and related files) |
| 7 | **Explore** | browse traces and videos on the finalized timeline |
| — | **Sync-free** (optional) | ellipse + Kerr next to the eye video, independent of `final_sync_df` |

After Kerr you typically have what Block Annotator and Event Explorer need (`final_sync_df.csv` + Kerr-annotated eye data).

## Block folder layout

```
path_to_animal_folder/
└── PV_106/
    └── 2025_09_04/
        └── block_015/
            ├── arena_videos/
            ├── eye_videos/
            │   ├── LE/   # video, timestamps.csv, DeepLabCut CSV
            │   └── RE/
            ├── oe_files/ # Open Ephys recording, or parsed_events.csv
            └── analysis/ # written by the GUI
```

This repository does not ship a pupil annotation model. Provide DeepLabCut-format `.csv` files (one per eye video), or convert your annotations to that format.

Custom sync: put `parsed_events.csv` under `oe_files/<recording>/` with timestamp columns (`Arena_TTL`, `L_eye_TTL`, `R_eye_TTL`) and matching `*_frame` columns.

See also [annotation.md](annotation.md) for reviewing the same block after preprocessing.
