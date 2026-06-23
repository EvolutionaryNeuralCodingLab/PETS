# Block Annotator outputs — context for plot-script agents

Condensed reference for building **small, standalone Python plotting scripts** that consume data marked with the **Block Annotator GUI** (`eye_tracking_system_tools.annotation.block_annotator`). Attach this file to a new Cursor agent.

---

## 1. What the annotator produces (primary input)

| Artifact | Location | Role |
|----------|----------|------|
| **Per-block annotation JSON** | `{output_folder}/{animal}_{date}_block_{num}_annotations.json` | **Main input** — list of user-marked events |
| **Config (optional)** | `{output_folder}/annotator_config.yaml` | Event type names, default ±100 ms window |
| **Block data (not in output folder)** | `block_path` field inside each JSON → block’s `analysis/` + `oe_files/` | Source traces for plotting snippets |

The annotator does **not** export waveforms. Plot scripts must **read annotations**, then **pull aligned data** from each event’s `block_path`.

**Optional secondary input:** Event Explorer **Export selection** writes `explorer_export_{timestamp}.npz` + sidecar `.json` (pre-aligned snippets, `time_rel_ms` with `timepoint_ms = 0`). Prefer annotator JSON + repo loaders when you need full control; use NPZ when reproducing Explorer exports only.

---

## 2. Annotation JSON schema (v1)

**Filename:** `{animal_call}_{experiment_date}_block_{block_num}_annotations.json`  
If `experiment_date` is null, omit the date segment: `{animal}_block_{num}_annotations.json`.

**Top-level fields:**

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | `1` | Bump if breaking |
| `animal_call` | str | e.g. `PV_106` |
| `experiment_date` | str \| null | `yyyy_mm_dd` |
| `block_num` | str | e.g. `015` |
| `block_path` | str | Absolute path to block folder |
| `sample_rate_hz` | float | OE sample rate for this block |
| `sync_source` | str | Usually `final_sync_df.csv` |
| `ms_axis_range` | `[min, max]` | Block timeline span in ms |
| `config_snapshot.event_types` | list[str] | Types available when saved |
| `events` | list[object] | All marked events (full replace on each save) |

**Each event object:**

| Field | Type | Meaning |
|-------|------|---------|
| `id` | str (UUID) | Stable event id |
| `event_type` | str | e.g. `saccade`, `blink`, `noise`, `pupil event` |
| `timepoint_ms` | float | Anchor time on master timeline |
| `start_ms` | float | Window start (default anchor − 100 ms) |
| `end_ms` | float | Window end (default anchor + 100 ms) |
| `range_half_width_ms` | float | Usually 100 |
| `row_index` | int \| null | Index into `final_sync_df` at mark time |
| `arena_frame` | int \| null | Arena video frame index |
| `l_eye_frame` | int \| null | Left eye frame index |
| `r_eye_frame` | int \| null | Right eye frame index |
| `note` | str | Free text |

**Time convention:** `timepoint_ms`, `start_ms`, and `end_ms` use the same clock as `final_sync_df`:

```text
ms_axis = Arena_TTL / (sample_rate_hz / 1000)
```

This matches zeroed Open Ephys time in `OERecording.get_data(start_time_ms, window_ms, ...)` (see `block_annotator/oe_streams.py` header comment).

---

## 3. Block folder layout (for pulling plot data)

From `block_path` in JSON:

```text
block_path/
  analysis/
    final_sync_df.csv       # required — master sync table
    left_eye_data.csv       # L eye (pipeline)
    right_eye_data.csv      # R eye (pipeline)
    le_df.csv / re_df.csv   # alternates
  oe_files/ ...             # Open Ephys — use OERecording
```

**`final_sync_df` required columns:** `Arena_TTL`, `Arena_frame`, `L_eye_frame`, `R_eye_frame`, `L_values`, `R_values`. Often also `ms_axis`.

**Invalid frame sentinels:** `Arena_frame` may be INT64_MIN (`≈ -9.22e18`) on bad rows; treat non-finite or huge negatives as missing (see `block_annotator/models.py` `_safe_frame`).

**Eye CSV columns (typical):** `center_x`, `center_y`, `width`, `height`, `phi`, `ms_axis`, frame column (`L_eye_frame` / `eye_frame`). Kerr/degree columns vary — inspect per study.

---

## 4. Repo code to reuse (do not reimplement)

| Need | Module |
|------|--------|
| Load one annotation file | `json` + path glob `*_annotations.json` |
| Parse events | `AnnotationEvent.from_dict` in `block_annotator/models.py` |
| Load sync table | `load_final_sync_df(block)` in `preprocessing/block_sync_core.py` |
| Block + OE handle | `BlockSync` in `preprocessing/BlockSync_class.py` or `block_annotator/block_loader.py` |
| OE snippets | `OERecording.get_data` / `get_analog_data` / `get_accel_data` |
| OE stream list | `block_annotator/oe_streams.py` → `list_oe_streams`, `load_decimated_trace` |
| Eye snippets (frame-first) | `event_explorer/snippet_extractor.py` → `extract_eye_snippet`, `extract_ep_snippet` |
| Catalog many JSONs | `event_explorer/catalog.py` → `build_catalog`, `discover_annotation_files` |
| Explorer NPZ sidecar | `event_explorer/export_io.py` |

**Environment:** `conda activate eye_repo_win` (or `eye_repo_linux`) then `pip install -e .` (see `README.md`).

---

## 5. Minimal loading patterns

### 5.1 Scan all events in an output folder

```python
import json
from pathlib import Path

def load_all_events(output_folder: Path) -> list[dict]:
    rows = []
    for path in sorted(output_folder.glob("*_annotations.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for ev in data.get("events", []):
            rows.append({**ev, "_annotation_file": str(path), **{k: data[k] for k in (
                "animal_call", "experiment_date", "block_num", "block_path", "sample_rate_hz"
            )}})
    return rows
```

### 5.2 Slice eye CSV on event window (ms)

```python
import pandas as pd

def slice_eye_ms(df: pd.DataFrame, start_ms: float, end_ms: float) -> pd.DataFrame:
    col = "ms_axis" if "ms_axis" in df.columns else None
    if col is None:
        raise ValueError("eye table has no ms_axis")
    m = (df[col] >= start_ms) & (df[col] <= end_ms)
    return df.loc[m].copy()
```

### 5.3 Align trials to event anchor (for rasters / averages)

```python
import numpy as np

def time_rel_ms(t_ms: np.ndarray, timepoint_ms: float) -> np.ndarray:
    return t_ms - timepoint_ms  # 0 = user mark
```

---

## 6. What plot scripts should look like

**Goals:** small, reviewable, one main figure per script; CLI with `argparse`; save PNG/PDF/SVG to user-chosen path.

**Suggested layout:**

```text
scripts/plots/
  plot_events_by_type.py      # counts / timeline overview
  plot_event_eye_snippet.py   # single event, L/R pupil or position
  plot_event_type_average.py  # mean ± SEM across events of one type
  README.md                   # one line per script + example command
```

**Conventions:**

- Use **matplotlib** (or pyqtgraph only if interactive is required).
- Accept `--output-folder`, `--annotation-json`, or `--events-csv` exported from catalog.
- Filter by `--event-type`, `--animal`, `--block`.
- Default window: `[start_ms, end_ms]` from JSON; optional `--half-window-ms` override.
- Log skipped events (missing block path, missing CSV, no OE) to stderr; do not crash whole batch.
- No GUI, no video I/O in plot scripts.
- Add `if __name__ == "__main__"` and document one example command in docstring.

**Good plot types for v1:**

- Event timeline (strip / rug) colored by `event_type`
- Per-event multi-panel: L pupil, R pupil, one EP channel — time aligned to `timepoint_ms`
- Per-`event_type` average ± SEM (N events in title)
- Histogram of inter-event intervals or counts by block

---

## 7. Explorer NPZ export (optional input)

If the user exported from **Event Explorer**:

| NPZ key | When |
|---------|------|
| `time_rel_ms` | Relative ms, anchor at 0 |
| `values` | Single-trial trace (single mode) |
| `trials`, `mean`, `sem` | Average mode |
| `stream_name` | `l_pupil`, `r_pupil`, `l_degrees`, `r_degrees`, `ep` |

Sidecar JSON lists `events[]`, `half_window_ms`, `normalization`, `alignment: "timepoint_ms=0"`.

```python
import json
import numpy as np

bundle = np.load("explorer_export_20260101_120000.npz", allow_pickle=True)
meta = json.loads(Path("explorer_export_20260101_120000.json").read_text())
t = bundle["time_rel_ms"]
y = bundle["mean"] if "mean" in bundle else bundle["values"]
```

---

## 8. Pitfalls

| Issue | Mitigation |
|-------|------------|
| Moved `block_path` | Re-resolve via `animal_call` / `date` / `block_num` or prompt user for remap |
| Stale eye CSV vs `final_sync_df` | Check mtimes; see `preprocessing/README.md` staleness guard |
| EP load cost | Downsample for overview; load only `[start_ms, end_ms]` per event |
| Empty event list | Exit 0 with message |
| Mixed event types in average | Filter `event_type` explicitly |

---

## 9. Agent prompt (copy-paste)

```
You are building small matplotlib plot scripts for PETS Block Annotator outputs.

Read ANNOTATOR_PLOT_SCRIPTS_AGENT_CONTEXT.md and the attached annotation JSON example.

Constraints:
- Input: *_annotations.json under the user's output folder; pull waveforms from block_path → analysis/ CSVs and oe_files via existing repo modules.
- Scripts live under scripts/plots/ (or user-specified); each script is focused, CLI-driven, saves figures to disk.
- Align time to timepoint_ms = 0 unless the script is an overview timeline.
- Reuse snippet_extractor / OERecording / load_final_sync_df; do not duplicate ms_axis ↔ OE alignment logic.
- Handle missing files gracefully; document example commands in each script and scripts/plots/README.md.
- No GUI, no video, no changes to block_annotator unless a tiny shared helper is justified.

Before coding many scripts, confirm with the user: which event types, which streams (L/R pupil, degrees, EP channel), and preferred output format (PNG dpi, single vs multi-page).
```

---

## 10. Files to attach to the agent

- `ANNOTATOR_PLOT_SCRIPTS_AGENT_CONTEXT.md` (this file)
- One real `*_annotations.json` from the user’s output folder
- `src/eye_tracking_system_tools/annotation/block_annotator/models.py`
- `src/eye_tracking_system_tools/annotation/event_explorer/snippet_extractor.py`
- `src/eye_tracking_system_tools/annotation/event_explorer/catalog.py`
- `src/eye_tracking_system_tools/preprocessing/block_sync_core.py` (load_final_sync_df)
- `src/eye_tracking_system_tools/preprocessing/OERecording.py` (get_data signature)

---

## 11. Related docs

- `README.md` — Block Annotator & Event Explorer launch
- `BLOCK_ANNOTATOR_IMPLEMENTATION_PLAN.md` — annotator design
- `EVENT_EXPLORER_IMPLEMENTATION_PLAN.md` — explorer design and export format
