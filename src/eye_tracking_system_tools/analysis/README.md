"""
Phase-1/2 analysis pipelines: analyzed blocks → event tables → figure pickles/PDFs.

All runs write under ``outputs/<run_name>/{figures,metadata}/``.
Empty ``--tag`` overwrites ``phase2_latest``; ``--tag my_label`` creates
``phase2_my_label``. Never write into ``figures/reproduction/``.

Quick start (local sample data)::

    PYTHONPATH=src python -m eye_tracking_system_tools.analysis \\
        --registry configs/sample_blocks.yaml \\
        --params configs/analysis_params.yaml \\
        --out-root outputs --tag sample

Paper lizard remount (Data-2)::

    PYTHONPATH=src python -m eye_tracking_system_tools.analysis \\
        --registry configs/paper_blocks.yaml \\
        --params configs/analysis_params.yaml \\
        --out-root outputs --tag paper_remount

Paper-faithful replot from frozen Fig-2j events (exact published event set;
Fig 2f comes from the archived nodowncast pickle)::

    PYTHONPATH=src python -m eye_tracking_system_tools.analysis \\
        --event-pickle src/eye_tracking_system_tools/figures/reproduction/main_figures/Fig_2_j/saccade_angles_data.pkl \\
        --params configs/analysis_params.yaml \\
        --out-root outputs --tag paper_events

Reproduction PDF baseline (copy into run ``figures/repro_baseline/``)::

    PYTHONPATH=src python -m eye_tracking_system_tools.analysis.repro_baseline \\
        --out-root outputs --tag paper_events

Flexible per-figure paper tool (block checklist before each panel, params
overrides, dated export): ``development/flexible_paper_figures_tool.ipynb``.
Catalog / GUI / export live in ``figure_catalog.py``, ``paper_gui.py``,
``paper_export.py``. Covers Fig 2c–2j, 1e (from a finalized jitter export),
and Fig 3a–3f. The all-at-once driver remains ``analysis_figure_suite.ipynb``.

Velocity / threshold tool — minimal **desktop Tk window**: both-eye speed
traces + draggable threshold (deg or px). Notebook:
``development/parameter_visualization_tool.ipynb``, or CLI::

    MPLCONFIGDIR=.mplconfig PYTHONPATH=src python -m eye_tracking_system_tools.analysis.param_tune_gui \
        --registry configs/mouse_M_002_blocks.yaml \
        --params configs/analysis_params_mouse.yaml

Drag the dashed line (or use the spinbox). **Auto-zoom** / **Full span** for
view. **Save thr → YAML** writes ``speed_threshold_deg_per_frame`` in degrees mode.

Jitter mounts — full interactive pipeline (browse blocks → tag modular/rigid/mouse
→ calibrate pixel size → pick epochs → histograms):
``development/jitter_mount_pipeline.ipynb``.
The widgets live in ``jitter_gui.py``; ``jitter_epochs.py`` stays headless.

Eye-movement span (tool 1) — browse blocks with eye CSVs → full-range + p5–p95
spans in degrees and pixels:
``development/eye_span_pipeline.ipynb``
(``JitterBlockBrowser(require="eye")`` + ``span_gui.SpanCharacteristicsPanel``;
headless: ``python -m eye_tracking_system_tools.analysis.eye_movement_span``).

Jitter from the CLI (registry must already list blocks)::

    # Interactive SpanSelector (desktop matplotlib window)
    PYTHONPATH=src python -m eye_tracking_system_tools.analysis.jitter_epochs \\
        --registry configs/jitter_mount_blocks.yaml \\
        --out-root outputs --tag "" --pick --plot

    # Or seed middle 50% non-interactively, then plot
    PYTHONPATH=src python -m eye_tracking_system_tools.analysis.jitter_epochs \\
        --registry configs/jitter_mount_blocks.yaml \\
        --out-root outputs --tag "" --seed-middle-frac 0.5 --plot

    # Pool a subset: --select prints a checklist of peak displacements,
    # --include names the blocks outright
    PYTHONPATH=src python -m eye_tracking_system_tools.analysis.jitter_epochs \\
        --plot --select
    PYTHONPATH=src python -m eye_tracking_system_tools.analysis.jitter_epochs \\
        --plot --include PV_143_block_001 PV_24_block_012

Whichever blocks end up pooled are listed with their per-block stats under
``blocks_used`` in ``metadata/jitter_pool_summary.yaml``; in the notebook the same
choice is a checkbox list (``jitter_gui.JitterPoolSelector``).

Section 7 of the jitter notebook maps pupil travel for those same blocks via
``eye_movement_span.py`` (p5–p95 hypot of ``center_x``/``center_y`` in px and µm,
plus Kerr degrees when present) and reports lizard/mouse ``jitter_p95 / eye_span``
ratios into ``metadata/eye_movement_spans.csv``.

Finalizing a comparison (``jitter_export.py``) freezes the current selection into
``outputs/jitter_comparison_figures_<tag>_<YYYYmmdd>_<HH>_<MM>/`` holding the PDFs,
``manifest.yaml``, ``epochs_table.csv`` and ``jitter_comparison_data.pickle``. The
pickle's ``epochs`` table is one row per pooled epoch — animal, date, block, mount,
eye, the frame span it was cut from and the µm/px factor applied — plus the
``frames`` and ``displacement`` arrays behind every plotted value::

    from eye_tracking_system_tools.analysis.jitter_export import (
        load_jitter_bundle, plot_from_bundle, bundle_samples_long,
    )

    bundle = load_jitter_bundle("outputs/jitter_comparison_figures_mount_v1_20260801_20_45")
    plot_from_bundle(bundle)                       # redraw as published
    plot_from_bundle(bundle, n_bins=30, xmax=400)  # restyle, same data
    plot_from_bundle(bundle, epochs=bundle["epochs"].query("eye == 'left_eye'"))
    bundle_samples_long(bundle)                    # one row per sample + frame index

Displacements default to µm: each eye's pixel trace is scaled by that block's
``analysis/LR_pix_size.csv`` (mm/px), and the run fails naming any block that lacks
it — ``--units px`` stays in raw pixels. ``pixel_calibration.py`` is a standalone
port of ``BlockSync.calibrate_pixel_size`` (drag an ROI whose diagonal spans a known
distance) that reads and writes the very same CSV without mounting a BlockSync.

Jitter *correction* itself (video cross-correlation, ``correct_jitter``) belongs to
the preprocessing GUI; this package only reads the resulting
``analysis/jitter_report_dict.pkl`` and quantifies residual camera movement.

Editable figure walkthrough: ``development/analysis_figure_suite.ipynb``.

Per-block saccade finalize (preprocessing GUI **Saccades** tab)
----------------------------------------------------------------
After Kerr angles exist, use the preprocessing GUI:

* **Calibration** tab — write ``analysis/LR_pix_size.csv`` (Qt landmark ROI or
  manual px lengths). Needed for pupil / jitter figures.
* **Saccades** tab — show L/R Euclidean velocity in **100 s windows** (Prev/Next),
  drag a speed threshold (GUI unit **deg/ms** → detector **deg/frame**), run
  detection on the full block, finalize under ``analysis/saccades/``::

       saccade_events.csv (+ .meta.yaml)
       saccade_synced.csv
       saccade_monocular.csv
       saccade_events.pkl          # full profiles for Fig 2c/2d
       detection_params.yaml
       detection_summary.yaml

``pipeline.build_event_tables(..., prefer_finalized=True)`` (default) loads those
files when present instead of re-detecting. Compile across a registry with
``development/compile_block_saccades.ipynb``; the flexible paper tool exposes
``USE_FINALIZED_SACCADES`` (default True). Helpers live in ``saccade_export.py``.

Mouse / pogona supplementary: point ``flexible_paper_figures_tool.ipynb`` at
``configs/mouse_M_002_blocks.yaml`` + ``configs/analysis_params_mouse.yaml``
(data under ``/Volumes/Data/Nimrod/experiments``). Mouse blocks have no
behavior-state files, so Fig 3c/3e/3f correctly show as ineligible.

Eye CSV selection prefers ``*raw_verified*`` (else newest) and always logs /
records the chosen left/right paths in ``*.meta.yaml`` sidecars.
Head labels use ``oe_files/**/lizMov.mat`` when present (paper notebook path).

Units: event ``peak_velocity`` is deg/frame (paper convention); Fig 2e converts
to deg/ms; Fig 2c traces use deg/sec. Fig 2c/2d default to ``plot_animals: [PV_106]``
with viridis amp-bin colors matching the archived paper pickle.
"""
