# Annotator plot scripts

CLI and GUI tools that read Block Annotator `*_annotations.json` files and pull waveforms from each event's `block_path`.

Requires the platform conda env (`eye_repo_win`, `eye_repo_linux`, or `eye_repo_mac`) and `pip install -e .` from the repo root. See [docs/install.md](../../docs/install.md).

## GUI

```bash
python scripts/plots/annotator_plot_gui.py
```

1. Choose the annotator **output folder** (contains `*_annotations.json`).
2. Select **blocks** and **event types**.
3. Choose streams: L/R pupil, L/R degrees, electrophysiology (HS channels).
4. Plot mode: average ± SEM, individual trials, or both.
5. Adjust window, figure size, fonts, DPI; export **main PDF** and **legend PDF** separately.

## CLI — pupil average

```bash
python scripts/plots/plot_event_type_pupil_average.py \
  --output-folder _annotator_out \
  --event-type saccade \
  --eye both \
  --half-window-ms 100 \
  --out _annotator_out/saccade_pupil_average.pdf \
  --legend-out _annotator_out/saccade_pupil_average_legend.pdf
```

## Shared module

`annotator_plot_core.py` loads annotations, normalizes time windows, and writes the matplotlib figures used by the GUI and the CLI.
