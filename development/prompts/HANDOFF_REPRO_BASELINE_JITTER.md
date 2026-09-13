# Handoff: Repro baseline, 2c/2d/2f fixes, jitter epoch pooling

**Plan file:** `/Users/nimi/.cursor/plans/repro_baseline_and_jitter_f476ed32.plan.md`  
**Repo:** `/Users/nimi/Projects/PETS`  
**Branch context:** `repro-analysis-pipelines` (do **not** commit unless the user asks)  
**Python env:** `/Users/nimi/miniforge3/envs/eye_repo_mac/bin/python` with `PYTHONPATH=src`

You are implementing the plan above. Read it fully before coding. Prior agent completed Phase 1–2 remount/mouse scaffolding; this pass is about **visual parity + jitter reporting**, not rebuilding detection from scratch.

---

## Goals (in order)

1. **Organized run outputs** — every new write under `outputs/<run_name>/{figures,metadata}/`. Default run name overwrites; `--tag <name>` creates a unique run. **Never** write into or overwrite files under `src/eye_tracking_system_tools/figures/reproduction/`.
2. **Baseline** — run existing reproduction `figure_*.py` / `fig_*.py`, **copy** PDFs to `figures/repro_baseline/` (+ manifest in `metadata/`).
3. **Fix analysis plots** — shared Okabe–Ito/`build_color_map`; fix 2c/2d (one animal default `PV_106`, mean bin traces, viridis like archived pickle); emit **2f** in paper-events mode from archived `Fig_2_f` pickle into the run’s `figures/`.
4. **Jitter** — extract amplitude traces from existing `jitter_report_dict.pkl` → manual epoch picker → pool samples → modular-vs-rigid histogram **and** a dedicated mouse histogram.

---

## Critical context (do not rediscover blindly)

### Why 2c/2d look wrong
[`figures_2c_2e._plot_pos_vel_pdfs`](src/eye_tracking_system_tools/analysis/figures_2c_2e.py) overlays **all animals**. Paper pickle [`Fig_2_c/pos_vel_by_amp_bins_bundle.pkl`](src/eye_tracking_system_tools/figures/reproduction/main_figures/Fig_2_c/pos_vel_by_amp_bins_bundle.pkl) has **only PV_106** with **5 mean amp-bin series**. Fix plotting, not (primarily) detection.

### Units (already fixed upstream)
- Event `peak_velocity` = **deg/frame** (matches Fig 2j pickle).
- Fig 2e converts to deg/ms via `× (fps/1000)`.
- Fig 2c traces = **deg/sec**.

### Why 2f was missing from `phase2_from_paper_events`
`--event-pickle` skips 2f (needs eye traces). Wire-through: plot archived [`Fig_2_f/figure_2f_nodowncast.pickle`](src/eye_tracking_system_tools/figures/reproduction/main_figures/Fig_2_f/figure_2f_nodowncast.pickle) into the **run** `figures/` folder (use helpers from `Fig_2_f/figure_2f.py`). Remount path still computes 2f live.

### Colors
Port notebook `COLOR_TEMPLATES` + `build_color_map` from `development/old_pipelines_for_ref/multiple_figures_pipeline_migration.ipynb` into `src/eye_tracking_system_tools/analysis/colors.py`.

### Jitter paradigm (user-corrected — follow this exactly)
1. Reports are **already** at `block/analysis/jitter_report_dict.pkl` → `{left_eye, right_eye}` with `top_correlation_dist` (frame-length amplitude), plus displacements.
2. **Extract** the amplitude trace for the picker (do not recompute from video).
3. User **manually selects stable epochs** on that trace (SpanSelector).
4. **Pool** all samples inside selected epochs across blocks.
5. Plot:
   - modular vs rigid comparison histogram (Fig 1e style; see `Fig_1_e/figure_1e.py`);
   - **separate** dedicated mouse histogram.
6. Registry: `configs/jitter_mount_blocks.yaml` with `mount_type: modular | rigid | mouse` (user fills paths).

Sample jitter dict verified on `/Users/nimi/sample_data/.../jitter_report_dict.pkl`.

---

## Existing entry points to extend (do not rebuild)

| Piece | Path |
|-------|------|
| CLI | `src/eye_tracking_system_tools/analysis/__main__.py` (`--registry`, `--event-pickle`, `--params`, `--out`) |
| Pipeline | `pipeline.py` (`run_from_registry`, `run_from_event_pickle`, `event_tables_from_saccade_angles_pickle`) |
| Params | `configs/analysis_params.yaml` |
| Paper registry | `configs/paper_blocks.yaml` |
| Mouse registry | `configs/mouse_M_002_blocks.yaml` |
| Editable notebook | `development/analysis_figure_suite.ipynb` |
| Prior outputs (flat; reorganize going forward) | `outputs/phase2_from_paper_events/`, `outputs/phase2_paper_figures/` |
| Backlog | `development/plans/PHASE2_REPRODUCTION_BACKLOG.md` |

Prefer adapting `--out` into `--out-root` + `--tag` (or resolve `out = out_root / run_name` with `figures/` + `metadata/` children) rather than scattering new top-level output folders.

---

## Suggested implementation checklist

- [ ] `resolve_run_dir(out_root, tag) -> (run_dir, figures_dir, metadata_dir)` + guard that path is not under `figures/reproduction`
- [ ] Baseline runner: execute repro scripts; **copy** PDFs → `figures/repro_baseline/`; manifest → `metadata/`
- [ ] `analysis/colors.py` + wire into 2e/2i/2j/2g
- [ ] Fix `_plot_pos_vel_pdfs` / export params: `plot_animals: [PV_106]`, viridis match
- [ ] Paper-events: write `figure_2f.pdf` into run `figures/` from archived pickle
- [ ] Re-run paper-events into organized run; short compare note in `metadata/`
- [ ] `jitter_epochs` module: load report → extract trace → picker → epoch YAML in `metadata/jitter_epochs/`
- [ ] Pool + `figures/jitter_modular_vs_rigid.pdf` + `figures/jitter_mouse.pdf`
- [ ] Update README + notebook sections; update backlog checkboxes
- [ ] Do **not** commit unless asked

---

## Constraints / preferences

- Prefer extracting from notebooks over cleaning them.
- Match repo style; no secrets; no huge data commits.
- `development/old_pipelines_for_ref/` may stay untracked — reference only.
- Faulty runs: default overwrite (`phase2_latest` / similar); tagged runs preserved.
- Reproduction tree is **read-only source of truth** for baseline pickles/scripts.

---

## Success criteria

- [ ] Baseline PDFs live under `outputs/<run>/figures/repro_baseline/` with reproduction originals untouched
- [ ] Analysis 2c/2d look like single-animal mean bin curves with paper-like viridis
- [ ] Paper-events run includes `figure_2f.pdf` in `figures/`
- [ ] Run dirs always split `figures/` vs `metadata/`; tag vs overwrite works
- [ ] Jitter: per-block epoch picking on extracted amplitude → pooled modular/rigid hist + mouse hist
- [ ] Handoff note / backlog updated for any remaining user-filled registry paths

---

## First commands for the implementing agent

```bash
export PATH=/Users/nimi/miniforge3/envs/eye_repo_mac/bin:$PATH
export MPLCONFIGDIR=/Users/nimi/Projects/PETS/.mplconfig
cd /Users/nimi/Projects/PETS

# Confirm mounts if remounting / jitter on server data
ls /Volumes/Data-2/Nimrod/experiments >/dev/null
ls /Volumes/Data/Nimrod/experiments/M_002 >/dev/null

# Read plan + current analysis README
# Then implement output-layout helper first, then baseline copy, then plot fixes, then jitter.
```

When done, summarize what was written where, how to tag a run, and what the user must fill in `configs/jitter_mount_blocks.yaml` before epoch picking.
