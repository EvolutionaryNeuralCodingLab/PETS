# Phase 2 backlog

## Done (this phase)
- [x] Paper cohort registry: `configs/paper_blocks.yaml` (23 blocks on Data-2)
- [x] Inventory + parity: `outputs/phase2_paper_cohort_inventory.md`,
  `outputs/phase2_paper_parity_report.md`
- [x] Full lizard remount → `outputs/phase2_paper_figures/`
- [x] R3-1 Fig 2f paper filters + S3 twin PDF (`figure_S3.pdf`)
- [x] `lizMov.mat` → `head_movement` labeling (`head_labels.py`)
- [x] Mouse M_002 mini Fig 2: `configs/mouse_M_002_blocks.yaml` →
  `outputs/phase2_mouse_M_002_figures/`
- [x] Starter lizard–mouse comparison PDFs:
  `outputs/phase2_lizard_mouse_comparison/`
- [x] Velocity-unit fix (deg/frame events; 2e→deg/ms; 2c→deg/sec)
- [x] Paper-events replot path (`--event-pickle`) →
  `outputs/phase2_from_paper_events/`
- [x] Editable suite notebook: `development/analysis_figure_suite.ipynb`
- [x] Organized run layout: `outputs/<run>/{figures,metadata}/` via
  `--out-root` + `--tag` (`run_layout.py`)
- [x] Reproduction baseline copy: `analysis.repro_baseline`
- [x] Shared Okabe–Ito colors (`analysis/colors.py`); 2c/2d single-animal
  + viridis; archived 2f in event-pickle mode
- [x] Jitter epoch extract/picker/pool (`analysis.jitter_epochs`) +
  `configs/jitter_mount_blocks.yaml` (user fills real modular/rigid/mouse paths)
- [x] Jitter notebook pipeline: ipywidgets block browser + inline epoch picker
  (`analysis/jitter_gui.py`, `development/jitter_mount_pipeline.ipynb`)
- [x] μm jitter histograms: standalone `analysis/pixel_calibration.py` (BlockSync-free
  port of `calibrate_pixel_size`, same `analysis/LR_pix_size.csv`) + notebook
  calibration panel; `pool_by_mount(units="um")` is now the default
- [x] Per-block pooling choice: checklist of blocks with mean/p95/max displacement
  (`JitterPoolSelector` widget, `run_plot_pooled(include=/select=)`, `--include`/
  `--select`); selection recorded as `blocks_used` in the pool summary
- [x] Finalized jitter deliverable (`analysis/jitter_export.py`): dated
  `jitter_comparison_figures_<tag>_<date>_<hh>_<mm>/` with PDFs, manifest,
  per-epoch table (animal/date/block/eye + frame span) and the frames/displacement
  arrays; `plot_from_bundle()` redraws or re-filters from the pickle alone
- [x] Flexible paper figures tool (`development/flexible_paper_figures_tool.ipynb`):
  per-figure block checklist + params overrides (`paper_gui.py`, `figure_catalog.py`),
  event-table cache (`event_cache.py`), dated export (`paper_export.py`); ports of
  Fig 3d / 3e–3f / 3a–3c into the analysis package; Fig 1e replots a finalized
  jitter export. Mouse supplementary = same notebook against
  `mouse_M_002_blocks.yaml` + `analysis_params_mouse.yaml`.

## Remaining scientific / reviewer follow-ups
- Remount still yields more events than Fig 2j (36084 vs 28146); use
  `--event-pickle` for exact published event-set plots. Dig into CSV /
  detection-pass differences if exact remount parity is required.
- Historical Fig 2e n=34876 ≠ 2j n=28146 (different export pass).
- Mouse QC: M_002 event rates look very high vs lizard; revisit Kerr/threshold
  / verified CSV choice before reviewer-facing claims.
- ISI split by active/quiet (`export_inter_saccade_intervals_brokenX_by_state`).
- Pupil state figures (3e/3f): analysis ports exist; run the flexible notebook
  against the full paper registry (21/23 blocks have behavior_state).
- Turtle: **out of scope until mouse demo accepted**.
- Populate the jitter registry with real modular / rigid / mouse blocks via
  `development/jitter_mount_pipeline.ipynb`, calibrate the blocks missing
  `LR_pix_size.csv`, complete epoch picking, then decide which blocks enter the
  pooled histograms (sample_data entries are placeholders).

## Tooling polish
- Explicit opt-in for `rotated_verified` eye CSVs (default remains `raw_verified`).
- Regenerate reproduction-folder pickles with full `*.meta.yaml` sidecars.
- Optional mouse frame-rate / amplitude-scale calibration pass.
