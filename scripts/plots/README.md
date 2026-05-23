# Annotator plot scripts

CLI and GUI tools that read Block Annotator `*_annotations.json` files and pull waveforms from each event's `block_path`.

Requires: `conda activate eye_annotator` and `pip install -e . --no-deps` from the repo root.

## GUI (recommended)

```bash
python scripts/plots/annotator_plot_gui.py
```

1. Choose the annotator **output folder** (contains `*_annotations.json`).
2. Select **blocks** and **event types**.
3. Choose streams: L/R pupil, L/R degrees, electrophysiology (HS channels).
4. Plot mode: average ± SEM, individual trials, or both.
5. Adjust window, figure size, fonts, DPI; export **main PDF** and **legend PDF** separately.

## CLI — pupil average only

```bash
python scripts/plots/plot_event_type_pupil_average.py \
  --output-folder _annotator_out \
  --event-type saccade \
  --eye both \
  --half-window-ms 100 \
  --out _annotator_out/saccade_pupil_average.pdf \
  --legend-out _annotator_out/saccade_pupil_average_legend.pdf
```

## Jupyter notebook (explore & edit)

Open `scripts/plots/annotator_plot_explorer.ipynb` in Jupyter or VS Code.

- Scan and inspect annotation exports as **pandas tables**
- Filter events with masks (type, block, animal, custom)
- Load **block caches** and inspect `final_sync_df` / eye CSVs / OE
- Extract snippets and plot inline; export PDFs when ready
- Copy stable cells into your own script

## Shared module

`annotator_plot_core.py` — loading, normalization, and matplotlib export used by GUI, CLI, and notebook.
