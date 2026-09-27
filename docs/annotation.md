# Block Annotator and Event Explorer

These apps run on a block that already has `analysis/final_sync_df.csv` (see [preprocessing.md](preprocessing.md)). Use the same conda env as the rest of the repository.

Worked example: **`PV_106 / 2025_09_04 / block_015`**.

## Block Annotator

Synchronized review of arena + eye videos, Open Ephys traces, and event marks.

```bash
python -m eye_tracking_system_tools.annotation.block_annotator \
  --block /path/to/PV_106/2025_09_04/block_015 \
  --output /path/to/annotator_output
```

Add `--dialog` for the setup dialog. Optional env vars: `PETS_ANNOTATOR_BLOCK`, `PETS_ANNOTATOR_OUTPUT`.

On first run, pick an output folder. If `annotator_config.yaml` is missing there, a template is created with default event types (`saccade`, `blink`, `noise`, `pupil event`). Then select a block folder that contains `analysis/final_sync_df.csv`.

Per-block output is `{animal}_{date}_block_{num}_annotations.json` in the output folder. Save replaces the full event list.

Headless check:

```bash
python scripts/validate_block_annotator_load.py --block /path/to/block_015 --output /path/to/annotator_output
```

## Event Explorer

Browse Block Annotator `*_annotations.json` events, inspect time-aligned eye and electrophysiology traces, and export NPZ.

```bash
python -m eye_tracking_system_tools.annotation.event_explorer --help
python -m eye_tracking_system_tools.annotation.event_explorer \
  --json /path/to/PV_106_2025_09_04_block_015_annotations.json
```

With no args, the app opens and prompts **Add sources…**. **File → Save session** writes `*.explorer_session.json`. **File → Export selection** writes `explorer_export_{timestamp}.npz` plus a sidecar JSON.
