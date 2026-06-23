# Saccade-aligned LFP average pipeline

Standalone tools for a quick per-animal check: average headstage (LFP) traces aligned to detected saccades, with **Concurrent**, **Monocular**, **Left**, **Right**, and **All** traces on each electrode panel.

Does not touch the preprocessing GUI or other branches.

## GUI (recommended)

```bash
python pipelines/saccade_lfp_average/saccade_lfp_gui.py
```

1. Choose **experiment folder** and **animal** — type paths directly or use **Browse…**.
2. Choose **analysis folder** (type or browse); exports go under `{analysis}/{YYYY-MM-DD}/`.
3. Enter an optional **plot label** (appended to the PDF filename).
4. Click **Scan blocks**, select blocks, enter HS channels and optional query.
5. Adjust parameters, then **Refresh plot**.
6. Toggle trace groups instantly without recomputing (Concurrent, Monocular, Left, Right, All).
7. **Export PDFs** — if the filename already exists in today's folder, a `_2`, `_3`, … suffix is added and you get a warning (no overwrite).

Embedded preview includes pan/zoom toolbar. Each panel legend shows **n** per trace (e.g. `Concurrent (n=132)`).

## Prerequisites

For each requested block:

1. Synchronization completed (`analysis/final_sync_df.csv` or `blocksync_df.csv`).
2. Eye tracking CSVs under `analysis/` — **`left_eye_data.csv` / `right_eye_data.csv`** (preferred) or **`le_df.csv` / `re_df.csv`** from sync / verification.
3. Open Ephys `.continuous` files under `oe_files/`.
4. Optional for filtering:
   - `analysis/block_XXX_behavior_state.csv` (`start_time`, `end_time`, `annotation` — values like `quiet` / `active`).
   - `lizMov.mat` under OE `analysis/` for the `accel` column.

Install the package from the repo root:

```bash
conda activate eye_repo_win   # or your env with PETS installed
pip install -e . --no-deps
```

## Usage

```bash
python pipelines/saccade_lfp_average/plot_saccade_lfp_average.py \
  --experiment-path D:/data/experiments \
  --animal PV_106 \
  --blocks 015 016 \
  --electrodes 1 2 3 4 \
  --out _out/PV_106_saccade_lfp.pdf
```

### Optional filters

Filter saccades with a pandas query on the event table columns:

| Column | Description |
|--------|-------------|
| `block` | Block number string |
| `eye` | `L` or `R` |
| `saccade_start_ms` | Saccade onset (ms, arena timebase) |
| `peak_velocity` | Peak pupil speed in the saccade |
| `accel` | Sum of accelerometer `movAll` around onset |
| `behavior` | Behavior annotation at onset (`quiet`, `active`, …) |
| `sync_status` | `synced` or `non_synced` (both detection modes) |
| `sync_pair_id` | Pair index for synced binocular saccades |
| `magnitude`, `angle` | Legacy mode saccade metrics |
| `saccade_on_ms` | Legacy onset time (same as `saccade_start_ms`) |

Example — synced saccades only (either mode):

```bash
--query "sync_status == 'synced'"
```

Example — saccades during low head movement:

```bash
python pipelines/saccade_lfp_average/plot_saccade_lfp_average.py \
  --experiment-path D:/data/experiments \
  --animal PV_106 \
  --blocks 015 \
  --electrodes 1 2 \
  --query "behavior == 'quiet'" \
  --out _out/PV_106_quiet_saccade_lfp.pdf
```

Example — combine accel and eye:

```bash
--query "accel < 50 and eye == 'L'"
```

### Window and detection

| Flag / GUI | Default | Meaning |
|------------|---------|---------|
| **Mode** | `velocity` | `velocity` = median + k×std on pupil speed; `legacy` = fixed `speed_r` threshold + L/R sync pairing |
| **Eye CSV source** | `auto` | Prefer `left/right_eye_data.csv`, else `le_df/re_df.csv` |
| `--half-window-ms` | `500` | ±500 ms around onset (1000 ms total) |
| `--saccade-threshold` | `2.0` | Velocity mode: threshold = median speed + k × std |
| `--min-saccade-frames` | `1` | Velocity mode: minimum high-speed run length |
| `--legacy-speed-threshold` | `2.0` | Legacy mode: `speed_r` cutoff |
| `--legacy-sync-diff-ms` | `680` | Binocular sync window (velocity and legacy modes) |
| `--legacy-magnitude-calib` | `1.0` | Legacy mode: magnitude scale |
| `--ep-noise-std-k` | `0` | Drop trials with snippet std > k × median(trial std); 0 = off |
| `--batch-size` | `200` | Saccades per `get_data` batch |

### EP noise filter

Per trial, the pipeline computes the **std of the aligned LFP snippet** (µV). Trials with  
`snippet_std > k × median(snippet_std)` are excluded before averaging (computed separately for each HS channel and eye group).

- **GUI:** set **Noise filter k** under Electrophysiology, then click **Apply** (re-averages cached trials without re-extracting).
- **CLI:** `--ep-noise-std-k 3`
- **0** disables the filter. Try **2–5** to drop single extreme traces; legend **n** counts trials after filtering.

| Flag | Default | Meaning |
|------|---------|---------|
| `--legend-out` | — | Separate legend PDF |
| `--figsize W H` | `2.4 1.8` | Size per electrode panel |
| `--dpi` | `300` | Output resolution |

## Output

- One PDF with **one subplot per electrode**.
- Each subplot can show up to **five traces** (toggle in GUI):

| Trace | Events included |
|-------|-----------------|
| **Concurrent** | Binocular synchronized pairs only (one LFP trial per pair, left-eye onset) |
| **Monocular** | All unpaired saccades from both eyes |
| **Left** | Unpaired saccades from the left eye only |
| **Right** | Unpaired saccades from the right eye only |
| **All** | Concurrent pairs + all monocular (L and R) |

- Vertical dotted line at saccade onset (t = 0).
- Y-axis: raw LFP in **µV** from `OERecording.get_data(..., convert_microvolts=True)`.

## Shared module

`saccade_lfp_core.py` — pipeline logic, plotting, and export used by both CLI and GUI.

## Notes

- **Velocity mode** (default): pupil speed segmentation + k×std threshold; events tagged synced/non_synced via binocular pairing.
- **Legacy mode**: `speed_r > threshold` on/off edges, magnitude/angle metrics, same sync pairing.
- GUI trace toggles: **Concurrent**, **Monocular**, **Left** (monocular L), **Right** (monocular R), **All** (concurrent + monocular).
- Eye data is loaded from `left/right_eye_data.csv` when present (newest `left_eye_data*.csv` glob), otherwise `le_df/re_df.csv`.
- Blocks missing prerequisites are skipped with a warning; the script fails only if no usable data remain.
- In the GUI, trace toggles redraw from cached averages; change detection/window/electrodes/query and click **Refresh plot** to recompute.
