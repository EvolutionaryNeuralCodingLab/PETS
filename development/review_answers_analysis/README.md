# Review-answer figures

Single-panel PDFs for the PLOS Biology revision. You assemble panels in the manuscript; these notebooks only write named PDFs plus CSV/NPY sidecars.

**Output root:** `outputs/review_answers_<tag>/`. Empty `TAG = ""` overwrites `outputs/review_answers_latest/`. Each plot is a self-contained folder:

```
outputs/review_answers_latest/
  metadata/event_cache/           ← local working cache (not copied with a plot)
  figure_2f/
    plots/                        ← PDFs
    metadata/                     ← pickle/CSV + params.yaml + cohort.yaml + LOGIC.md
    replot.py                     ← redraw PDFs into plots/replot/ (no PETS install)
  mouse_figure_2f/
  jitter_turtle/
```

A reviewer copies one plot folder and runs `python replot.py`. That redraws from metadata only; it does not re-detect saccades.

Do **not** copy the three existing tools. Run them in place; they now write plot folders under their run root (`jitter_latest/jitter_turtle/`, etc.). Collect with:

```bash
python -m eye_tracking_system_tools.analysis.review_collect \
  outputs/paper_latest outputs/jitter_latest outputs/yield_latest \
  outputs/review_answers_latest
```

Kernel: `eye_repo_mac`. From the repo root, `PYTHONPATH=src`.

---

## Tutorial: how to run this module

1. Confirm the lab volume is mounted and that every path in the registry YAML exists as written. Missing data should fail so you can edit the YAML; there is no volume aliasing.
2. Open a notebook below, set `TAG` if you want a timestamped snapshot, and run all cells. Each notebook:
   - loads a **block registry** YAML (`configs/paper_blocks.yaml` unless noted)
   - loads **analysis params** (`configs/analysis_params.yaml` unless noted)
   - builds or reuses an **event cache** under that run’s `metadata/event_cache/`
   - writes each figure into `outputs/review_answers_latest/<plot_id>/{plots,metadata}/` plus `replot.py`
3. The event cache stays under the run’s `metadata/event_cache/` (shared local working files, not part of a portable plot folder). Per-block raw files stay on disk under each block’s `analysis/` folder (see [Block registries](#block-registries--which-file-goes-with-which-block)).
4. After the three existing tools and the six notebooks have run, collect a clean set:

```bash
python -m eye_tracking_system_tools.analysis.review_collect \
  outputs/paper_latest outputs/jitter_latest outputs/yield_latest \
  outputs/review_answers_latest
```

Library code lives under `src/eye_tracking_system_tools/analysis/`. Notebooks are thin wrappers around `export_*` functions.

### Layout

```
development/review_answers_analysis/
  README.md                         ← this file
  species_traces.ipynb
  saccade_robustness.ipynb
  state_isi.ipynb
  qc_reversals_noise_blinks.ipynb
  corrective_head_timing.ipynb
  s1_occupancy.ipynb
  2f_figure_edits.ipynb
  figure_2e_edits.ipynb

outputs/review_answers_latest/
  <plot_id>/plots/*.pdf
  <plot_id>/metadata/             ← pickle/CSV + LOGIC.md + cohort.yaml
  <plot_id>/replot.py
  metadata/event_cache/<sha1>.pkl
```

---

## Block registries — which file goes with which block

| Cohort | Registry YAML | Params YAML | Animals / blocks |
|---|---|---|---|
| Lizard paper | [`configs/paper_blocks.yaml`](../../configs/paper_blocks.yaml) | [`configs/analysis_params.yaml`](../../configs/analysis_params.yaml) | PV_106 blocks 008–012; PV_143 001–004; PV_62 024, 026, 038; PV_126 007–012; PV_57 007–009, 012–013 (23 blocks) |
| Mouse mini-Fig 2 | [`configs/mouse_M_002_blocks.yaml`](../../configs/mouse_M_002_blocks.yaml) | [`configs/analysis_params_mouse.yaml`](../../configs/analysis_params_mouse.yaml) | M_002 blocks 012–015 |
| Jitter (R2-2, R3-10) | [`configs/jitter_mount_blocks.yaml`](../../configs/jitter_mount_blocks.yaml) | (jitter notebook) | modular vs rigid lizard mounts; mouse; turtle |
| Data yield (R2-1, R2-4) | [`configs/data_yield_blocks.yaml`](../../configs/data_yield_blocks.yaml) | (yield notebook) | same grouping as jitter |
| Species traces | hardcoded in `species_traces.DEFAULTS` | — | lizard `PV_126` block_007; mouse `M_002` block_012; turtle `T_18` block_001 |

On-disk per-block files (not copied into `metadata/`):

| File under `{block_path}/` | Used for |
|---|---|
| `analysis/left_eye_data*raw_verified*.csv` (and right) | φ/θ traces (S1, 2c/2d, 2f, QC, traces) |
| `analysis/block_NNN_behavior_state.csv` | active/quiet epochs (ISI, reversals, noise hist) |
| `analysis/saccades/` | finalized events if present (event cache prefers these) |
| `oe_files/**/lizMov.mat` | head-movement labels and peri-saccade timing |
| `analysis/noise_epochs_*.csv` | nictitating / pupil-perimeter QC |
| DeepLabCut CSVs in the eye video folders | likelihood fallback for nictitating midpoints |

Review-run `metadata/` files are **cohort-wide**, not one-per-block. The CSV `animal` / `block` columns identify the source row.

---

## Existing notebooks (run in `development/`)

| Notebook | Reviewer IDs | How the figure is made | PDFs |
|---|---|---|---|
| [jitter_mount_pipeline.ipynb](../jitter_mount_pipeline.ipynb) | R2-2, R3-10 | Mount-jitter epochs → µm at the eye plane | folders `jitter_modular_vs_rigid/`, `jitter_mouse/`, `jitter_turtle/` (PDFs inside `plots/`) |
| [data_yield_report_tool.ipynb](../data_yield_report_tool.ipynb) | R2-1, R2-4 | DLC pupil likelihood + ellipse completeness after noise/manual removals | folders `yield_dlc_likelihood_*/`, `yield_ellipse_*/` |
| [flexible_paper_figures_tool.ipynb](../flexible_paper_figures_tool.ipynb) | R3-1, R2-5, R1-6/7, R3-13 | Catalog runners (`2b`, `2c`/`2d`, `2e`, `2f`, `2g`, `2h`, `3a`/`3b`, `s3`) | **Lizard:** `figure_2f/plots/figure_2f.pdf` (+ colorbar), same pattern for 2b/2c_2d/2e/2g/2h/3a/3b/S3. **Mouse:** folder `mouse_figure_*` with the same unprefixed PDF names so lizard files are not overwritten. QC mouse speed threshold before export. |

Turtle jitter/yield/traces are system-works panels only (no locomotion/state claim).

Fig 2e scatters (`figure_2e_all_events_scatter.pdf`, `_concurrent_scatter.pdf`, `_monocular_scatter.pdf`) use **one animal** (`scatter_animal: PV_106` in the lizard YAML; `M_002` for mouse) and **the same xlim/ylim** as the multi-animal means panel.

---

## New notebooks (this folder)

| Notebook | Reviewer IDs | How the figure is made | PDFs / metadata |
|---|---|---|---|
| [species_traces.ipynb](species_traces.ipynb) | R2-1, R2-4 | φ/θ from one block per species, recentered, independent y-limits | folders `trace_lizard/`, `trace_mouse/`, `trace_turtle/` |
| [saccade_robustness.ipynb](saccade_robustness.ipynb) | R3-2 | See [R3-2](#r3-2-tracking-noise-vs-the-shuffle) | folder `robustness/` (threshold, window, noise-floor, optional shuffle + CSVs) |
| [state_isi.ipynb](state_isi.ipynb) | R3-18, R1-8 | Same-epoch ISIs (drop intervals that cross a state boundary); epoch lengths on a log-x histogram | folders `ISI_by_state/` and `epoch_durations/` |
| [qc_reversals_noise_blinks.ipynb](qc_reversals_noise_blinks.ipynb) | R3-5, R3-7, R3-14, R3-21 | Reloads traces; reversal = ≥50% return toward start within 80 ms after offset | folder `qc_reversals_noise_blinks/` |
| [corrective_head_timing.ipynb](corrective_head_timing.ipynb) | R3-15, R3-16 | Corrective = contra monocular partner in (0, 80] ms and ≤45°; head peri from `lizMov.mat` rising edges | folder `corrective_head/` |
| [s1_occupancy.ipynb](s1_occupancy.ipynb) | R3-4 | Likelihood histogram of Euclidean eccentricity `hypot(k_phi, k_theta)`; line at ±35°. Filename kept as `s1_empirical_ratio_hist.pdf` | folder `s1_occupancy/` |
| [mouse_2c_prepeak.ipynb](mouse_2c_prepeak.ipynb) | mouse 2c QC | Pre-peak speed bump: neighbor ISI, split averages, NaN-mask vs drop isolation. Does not change the production 2c exporter. | folder `mouse_2c_prepeak/` (`figure_2c.pdf`, `figure_2c_isolated.pdf`) |
| [2f_figure_edits.ipynb](2f_figure_edits.ipynb) | Fig 2f colormap trials | Lizard 2f (head-stationary), mouse 2f, lizard S3 — each in `paper_white0`, `turbo`, `hot`, and `hot_r` | run root `outputs/2f_colormap_trials/` with bundles `figure_2f/`, `mouse_figure_2f/`, `figure_S3/` |
| [figure_2e_edits.ipynb](figure_2e_edits.ipynb) | Fig 2e short-event ridge | Length-colored / gray scatters, all–concurrent–monocular means triptych, omission_trial (drop length ≤ 2); identity-line occupancy table | folder `diagnostics_2e/` (+ nested `omission_trial/`) |

---

## R3-2: tracking noise vs the shuffle

Eye-tracking papers usually **do not** shuffle saccade onsets to measure noise. Standard approaches (Engbert & Kliegl 2003; Nyström & Holmqvist 2010) estimate a **velocity noise floor** from samples that are not saccades:

1. Angular step per frame `hypot(Δφ, Δθ)` on every inter-sample interval.
2. Mask out saccade windows (here: event on–off plus 40 ms padding).
3. Robust scale: MAD of the remaining steps, converted to a Gaussian-equivalent σ (`MAD / 0.6745`). The detector threshold is then a multiple of that σ (Engbert’s λσ rule). RMS of successive samples (RMS-S2S) is the other common precision number.

`robustness_noise_floor.pdf` is that histogram, with the 0.8°/frame detector line and the cohort median MAD-σ. `robustness_noise_floor.csv` is per eye/block. Threshold and pairing-window sweeps stay as fast line plots (`robustness_threshold_sweep.pdf`, `robustness_window_sweep.pdf`).

### What the shuffle is (and is not)

The circular-shift shuffle is a **binocular coincidence null**, not a tracking-noise test.

- Within each block, left-eye onset times stay put. Right-eye onsets are shifted by a random offset and wrapped around that block’s time span, then re-paired with the same 34 ms window.
- Repeating this (~200 times; default — set `n_shuffle=0` to skip) gives a null distribution of the **monocular fraction**.
- Observed vs null asks: given these event *rates*, is L/R pairing tighter than chance?

It does **not** say anything about ellipse jitter, pixel noise, or whether 0.8°/frame is conservative. A naive permutation of the same R onset *values* among R rows does not change coincidence counts (same multiset of times), which is why the code circular-shifts in time instead.

The old implementation re-ran pandas `find_synced_saccades_ms` 1000 times and was too slow to be useful. Pairing is now numpy on onset arrays; 200 shifts are enough for a histogram.

---

## Mouse Fig 2c / 2d parameters

From `configs/analysis_params_mouse.yaml` → `main_sequence`:

| Key | Value | Meaning |
|---|---|---|
| `align_to` | `peak` | t = 0 is time of peak unsigned angular speed |
| `dt_ms` | 2 | occupancy grid |
| `bandwidth_ms` | 4 | Gaussian σ on that grid, in ms (`σ_bins = bandwidth_ms / dt_ms` = 2). Was 10 ms (σ = 5 bins), which made mouse traces look too smooth |
| `smoothing` | true | smooth numerator and occupancy separately, then divide |
| `velocity_unit` | `deg/sec` | plotted speed |
| `position_calc` | `mov_axis` | Fig 2c is **signed** displacement along the movement axis (can sit slightly below 0 around baseline) |

Raw speed is `hypot(Δφ, Δθ) / Δt` (never negative). Values below 0 on the **speed** panel were Gaussian undershoot on sparse occupancy; the exporter now clips speed at 0.

---

## Filters (locked)

- **Fig 2f:** `event_mode=all`, head-stationary, equal-animal weights. Concurrent L/R pairs (`binocular.sync_diff_ms`, 34 ms) contribute **one** point from both detected event peaks. Unpaired (monocular) events all stay in: ipsilateral event peak vs contralateral ±51 ms window sample (so true mono sits near the axis, near-threshold contra motion sits off-axis). Macro/micro share vmax. Not saved as S3.
- **Fig S3:** same collector as 2f, all saccades, still vs moving, **shared** colorbar between those two PDFs, not shared with 2f.
- **Fig 2e class split:** binocular pairing 34 ms; scatters = one animal, shared axes with the means panel.
- **ISI by state:** same bins as Fig 3d; drop ISIs that cross epoch boundaries.
- **Reversals:** trace-return ≥50% toward start within 80 ms after offset; split by active/quiet at onset.
- **Corrective:** contralateral monocular partner in (0, 80] ms and within 45° of the first movement axis; bars are non-corrective vs corrective, ylim 100%.
- **Head timing:** `lizMov.mat` rising edges; `saccade_on − nearest_head_onset` in ±250 ms for `head_movement==True` (events are re-labeled from lizMov at export).
- **S1:** empirical eccentricity occupancy only (no Blender); vertical line at published ±35°.

Library code lives under `src/eye_tracking_system_tools/analysis/`.
