# Preprocessing GUI — acceptance checklist

This document records what the **Preprocessing GUI** covers relative to the five
preprocessing notebooks, and what remains outside the GUI or requires external tools.

## Covered in the GUI (raw block + DLC → downstream-ready)

| Stage | Notebook | GUI tab | Primary outputs under `block_xxx/analysis/` |
|-------|----------|---------|-----------------------------------------------|
| 1 — Sync | `block_synchronization.ipynb` | **Sync** | `final_sync_df.csv`, `left_eye_data.csv`, `right_eye_data.csv`, brightness/sync intermediates |
| 2 — Verify | `data_verification.ipynb` | **Verify** | `self_kerr_refs.csv`, corrected `left/right_eye_data.csv` |
| 3 — Kerr | `kerr_degree_conversion.ipynb` | **Kerr** | `left/right_kerr_angle_<tag>.csv`, `left/right_eye_data_<tag>.csv` |
| 4 — Behavior | `add_accelerometer_state_annotations.ipynb` | **Behavior** | `block_<NNN>_behavior_state.csv` (when `lizMov.mat` exists) |
| 5 — Sync-free (optional) | `sync_free_eye_ellipse_pipeline.ipynb` | **Sync-free** | Per-eye `*_eye_data.csv` + `*_kerr_refs.csv` + `*_meta.json` beside videos; optional `analysis/*_timeline.csv` |

**Typical downstream path (sync-based):** complete Sync → Verify → Kerr. Export merged
`left/right_eye_data_<tag>.csv` with Kerr angles and use `final_sync_df.csv` for alignment.

**Optional:** Behavior tab for accelerometer state segments; Sync-free tab when you want
per-video ellipse/Kerr artifacts without repeating the main sync ellipse export.

## Prerequisites (user-provided, not produced by GUI)

- Organized `block_xxx/` folder layout per `BlockSync` (arena + eye videos, `oe_files/`, DLC CSV per eye).
- DeepLabCut (or compatible) pupil CSV in each eye folder.
- Open Ephys recordings **or** a hand-built `parsed_events.csv` with TTL columns.

## External / out of scope for v1

| Item | Notes |
|------|--------|
| `manual_outlier_annotation.ipynb` | Not in v1 GUI |
| MATLAB `getLizMovement` | Required once per block to create `lizMov.mat` for Behavior tab |
| Pupil DLC model training | User supplies DLC exports |
| Block Annotator / Event Explorer | Separate downstream annotation tools (same `eye_repo` env) |

## Known GUI gaps vs original plan

- **Sync tab 1.1:** no in-tab `channeldict` / `bad_blocks` editor (`PreprocConfig` only).
- **Golden regression snapshots:** `tests/golden/preprocessing_gui/` not populated.

## Sync-free artifact model (unified)

After **Finalize** on the Sync-free tab, each eye folder contains exactly:

1. `{left|right}_syncfree_<tag>_kerr_refs.csv`
2. `{left|right}_syncfree_<tag>_eye_data.csv` — one row per video frame (ellipse + Kerr)
3. `{left|right}_syncfree_<tag>_meta.json`

Optional: `analysis/{left|right}_eye_syncfree_<tag>_timeline.csv` when timeline mapping is enabled.

Draft ellipses (`*_draft.csv`) exist only between **Run ellipses** and **Finalize**.

## Self-check commands

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
pytest tests/test_preprocessing_gui_*.py -q
python scripts/validate_preprocessing_gui_load.py `
  --experiment-path D:\sample_data_for_eye_repo --animal PV_106 --block 015
```

Launch:

```powershell
conda activate eye_repo
python -m eye_tracking_system_tools.annotation.preprocessing_gui `
  --experiment-path D:\sample_data_for_eye_repo --animal PV_106 --block 015
```
