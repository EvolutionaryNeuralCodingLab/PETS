# Forward-facing split (messy vs public)

This file lives only on `revisions_messy`. It records what the public branch
(`forward-facing` → `main`) **does not** include, and what still has to be
frozen before the paper plot bundle is complete.

Local `outputs/` (~3.5 GB) was never committed. It remains on disk in the
project folder and is gitignored.

## On the public branch (intended `main`)

- `src/eye_tracking_system_tools/preprocessing/` (including the six notebooks already on `main`)
- `src/eye_tracking_system_tools/annotation/` (Block Annotator, Event Explorer, Preprocessing GUI)
- `src/eye_tracking_system_tools/analysis/` (library + analysis GUIs; not the S-figure entry point)
- `src/eye_tracking_system_tools/figures/reproduction/main_figures/` (unchanged from `main`)
- `src/eye_tracking_system_tools/figures/reproduction/supplementary/` (S1, S3, S8–S13 pickle + `figure_S*.py`)
- `src/eye_tracking_system_tools/raspberry_pi/`, `utils/`
- curated `configs/`, setup `scripts/`, `tests/` (except LFP tests)
- root README / setup / env files

## Excluded from public `main` (kept here)

### Construction workspaces

- `development/` (this folder): plans, prompts, scratch, `old_pipelines_for_ref/`,
  `review_answers_analysis/` notebooks, GUI-wrapper notebooks, nystagmus plot dumps,
  Kerr-error exploration, DLC QC notebook
- `src/.../figures/reproduction/video_creation/video_exporter_new.ipynb`
  (`synchronized_video_creation.ipynb` stays; it is already on `main`)

### Off-main pipelines / GUIs (to bring up to Preprocessing GUI standard later)

- `pipelines/saccade_lfp_average/` (saccade-triggered LFP GUI)
- `tests/test_saccade_lfp_detection.py`
- `scripts/plots/annotator_plot_explorer.ipynb`
- `scripts/measure_eye_size_on_sensor.py`, `scripts/overall_jitter_rejection_stats.py`
- `scripts/run_video_exporter_smoke.py`

The analysis *library* modules (`eye_size_on_sensor.py`, `eye_size_gui.py`,
`dlc_validation/`) remain in the package so existing unit tests still run.
They are **not** documented as first-class public GUIs.

### Lab-only / aggregator scripts

- `scripts/copy_offline_blocks_ab.py`, `scripts/copy_offline_blocks_ab_slim.py`
- `scripts/aggregate_supp_pdf_materials.py`, `scripts/export_material_agg_pdf_corrected.py`
- `configs/paper_blocks_custom.yaml`, `configs/paper_blocks_dryrun_PV_143.yaml`
- `occlusion_limited_eye_range_table.csv`

### Not in git at all

- `outputs/` (all run trees, event caches, Illustrator packs)
- `.cursor/`, matplotlib caches

## Supplementary figures — shipped vs still open

Canonical source copied into the public tree:
`outputs/material_agg_pdf_corrected_FINAL/` (local, gitignored).

| Figure | Public status | Notes |
|---|---|---|
| S1b, S1c | Shipped | `kerr_component_error.pkl` + `figure_S1.py` |
| S1a | PDF shipped, **not replottable** | Static Blender crop from Document S2; no pickle |
| S2 | **Missing** | No plot-bundle in the agg pack |
| S3 | Shipped | `figure_S3.pickle` + `figure_S3.py` |
| S4–S7 | **Missing** | No plot-bundles in the agg pack |
| S8b,d,f | Shipped | species-trace pickles |
| S8a,c,e | PNG shipped, **not replottable** | Eye-video stills; no pickle |
| S8g,h | Shipped | nested `mouse_2c_2d_isolated` pickle |
| S8i | Shipped | nested `mouse_figure_2e` pickle |
| S8j | Shipped | nested `S8j` lizard/mouse 2f pickle |
| S9 | Shipped | `unified_jitter.pkl` + `figure_S9.py` |
| S10 | Shipped | three Rayleigh pickles + `figure_S10.py` |
| S11 | Shipped | epoch-trial + ISI + raw-epoch pickles + `figure_S11.py` |
| S12a | Shipped | `double_steps_move_only.pkl` + `figure_S12.py` |
| S12b | PDF shipped, **not replottable** | Example trace; no pickle |
| S13 | Shipped | `s13_preonset_2e.pkl` + `figure_S13.py` |
| s1_occupancy (R3-4) | **Not in agg pack** | CSV-only bundle under `outputs/replotting_standalone/s1_occupancy/` |

## Finalization still required (before the paper plot bundle is complete)

1. Identify S2 and S4–S7 (manuscript vs video vs table vs figure) and freeze pickle+script folders if they are plots.
2. Replace S1a with a pickle-backed Blender/occupancy export, or document it as a static illustration.
3. Freeze S12b example traces into a pickle, or document the example as a static PDF.
4. Decide whether R3-4 `s1_occupancy` belongs in the public S1 folder.
5. Optional: license file; saccade-LFP GUI once it matches Preprocessing GUI quality.
6. Optional: drop undocumented analysis GUIs (`eye_size_gui`, DLC validation) from `src/` or promote them.
