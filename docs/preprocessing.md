# Preprocessing GUI

The Preprocessing GUI is the way to process an acquired block. It is a tabbed PyQt6 app on a `block_xxx/` folder.

```bash
conda activate eye_repo_mac   # or eye_repo_win / eye_repo_linux
python -m eye_tracking_system_tools.annotation.preprocessing_gui
```

Add `--dialog` to pick experiment, animal, and blocks interactively. To open one block without that dialog, pass the folder that contains the animal folder, the animal, and the block:

```bash
python -m eye_tracking_system_tools.annotation.preprocessing_gui \
  --experiment-path /path/to/folder_that_contains_PV_106 \
  --animal PV_106 \
  --block 015
```

`--experiment-path` is the parent of `PV_106`. Session prefs (`preproc_gui_config.yaml`) default to the repository `configs/` folder and are updated on close. Override with `--output /path/to/gui_output` or `PETS_PREPROC_OUTPUT` if you need a different location.

Worked example (shortest analyzed block): **`PV_106 / 2025_09_04 / block_015`**. It is not stored in this software repository. Download `PV_106.zip` from the associated dataset ([record](https://zenodo.org/records/23016364), DOI [10.5281/zenodo.23016364](https://doi.org/10.5281/zenodo.23016364)) and unzip it. That zip contains the example block, including the videos and Open Ephys files. The example already includes `lizMov.mat`, so the Behavior tab can load it. It does not include an accelerometer calibration file, so **Compute lizMov (Python)** uses built-in defaults and says so in the terminal. If the last Open Ephys record is a few samples short, that command prints whether only record padding was missing or whether the end of the accelerometer window was left as zeros, and what that means for eye/arena synchronization versus head-movement timing.

## Recommended tab order

| Step | Tab | Result |
|------|-----|--------|
| 1 | **Sync** | `analysis/final_sync_df.csv`, then `left_eye_data.csv` / `right_eye_data.csv` |
| 2 | **Verify** | overlays, Kerr refs (`self_kerr_refs.csv`), corrected eye CSVs |
| — | **Refine** (optional) | image-based ellipse refinement sidecars (`*_eye_data_refined_<tag>.csv`) |
| 3 | **Conicoid** | gaze normal `c_nx/c_ny/c_nz` in `left/right_conicoid_angle_<tag>.csv`, merged `*_eye_data_degrees_<tag>_conicoid.csv`, `conicoid_fit_meta_<tag>.json` |
| 4 | **Kerr** | `left/right_kerr_angle_<tag>.csv` and merged eye CSVs |
| 5 | **Conicoid → Compare with Kerr** | `conicoid_vs_kerr_<tag>.csv`, `conicoid_vs_kerr_summary_<tag>.csv`, `conicoid_vs_kerr_<tag>.png` |
| 6 | **Calibration** | `analysis/LR_pix_size.csv` (needed for pupil / jitter) |
| 7 | **Behavior** (optional) | `block_<NNN>_behavior_state.csv` from accelerometer / `lizMov.mat` |
| 8 | **Saccades** | `analysis/saccades/` (`saccade_events.csv` and related files) |
| 9 | **Explore** | browse traces and videos on the finalized timeline |
| — | **Sync-free** (optional) | ellipse + Kerr next to the eye video, independent of `final_sync_df` |

After Kerr you typically have what Block Annotator and Event Explorer need (`final_sync_df.csv` + Kerr-annotated eye data). Those tools, and Saccades and Explore, still read Kerr's `k_phi` / `k_theta`; the conicoid output does not change them.

### Conicoid and the Kerr check

Conicoid (Swirski's pupil-on-sphere fit) is the gaze analysis to trust. Kerr stays in the pipeline as a sanity check and is not overwritten. Run both with the same `name_tag`, then press **Compare with Kerr** in the Conicoid tab (the batch run does this automatically when Kerr outputs exist).

- **Gaze result** is the unit normal `c_nx/c_ny/c_nz`. `c_phi/c_theta` are the same direction in Kerr's nested chart, kept for old consumers.
- **Disagreement** is the 3D angle between the conicoid normal and the normal implied by Kerr's `(phi, theta)`. Pairs are never subtracted. The summary lists the median, 90th percentile, and the fraction of frames that stayed finite for each method, per eye.
- **Ellipse geometry** must be version 2 (`analysis/ellipse_geometry_meta.json`, written by current Sync/Verify). For older blocks, the repaired `analysis/ellipse_v2/` tables from `python -m eye_tracking_system_tools.preprocessing.ellipse_recovery` are used instead. Otherwise the tab refuses to run.
- **Principal point** is the per-eye Kerr reference from Verify (`self_kerr_refs.csv`); the image centre is used, with a warning, when it is missing.
- **Focal length** comes from `analysis/LR_focal_length.csv` (`L_focal_length`, `R_focal_length`, in pixels), else the documented 652.6 px fallback (QILENS 3.7 mm on OV5647, scaled to 640 px). The source is recorded in `conicoid_fit_meta_<tag>.json` and in the comparison summary. Calibration's `LR_pix_size.csv` is not used: it is millimetres per pixel at the eye, not the sensor pixel pitch, so it does not give a focal length.
- **Sphere method** is Swirski 2D with RANSAC on. Dierkes 3D is selectable but provisional: it has not been validated against ground truth.

`examples/compare_kerr_conicoid_ground_truth.py` scores both methods against the synthetic Blender grid in `development/kerr_relative_error/` with the same 3D-angle function.

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