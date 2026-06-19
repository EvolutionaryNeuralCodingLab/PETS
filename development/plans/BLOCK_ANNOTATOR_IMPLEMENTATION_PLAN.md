# Block Annotator GUI — Implementation Plan & Agent Prompt

This document is the single source of truth for building the **Block Annotator**: a standalone PyQt6 GUI for synchronized multi-stream review and event annotation on PETS blocks with `final_sync_df.csv`.

---

## 1. Product summary

| Item | Decision |
|------|----------|
| Purpose | Parallel workflow tool (not replacing `manual_outlier_annotation.ipynb` yet); basis for future extensions |
| Input | User picks **output folder** + **block folder** (must contain `analysis/final_sync_df.csv`) |
| Master timeline | All rows in `final_sync_df`, with `ms_axis = Arena_TTL / (sample_rate/1000)` (compute on load if column missing) |
| Sync | Videos + traces + annotations share one scrub index `i` ∈ `[0, n-1]` and `ms_axis[i]` |
| Tech | **PyQt6**, **OpenCV** for video, **pyqtgraph** for traces (add dependency), reuse patterns from `GUI_example.py` (playback only) |
| Reference | `GUI_example.py` — keep video/transport ideas; drop all VisPy/3D |

---

## 2. User requirements (locked)

### 2.1 Config & event types

- On startup user provides **output folder** (required).
- **Config file** optional; if missing, auto-create template in output folder:
  - Default event types: `saccade`, `blink`, `noise`, `pupil event`
  - User-editable list (names + optional color/shortcut fields for future use)
- Config path can be chosen via file dialog or default `{output_folder}/annotator_config.yaml`

### 2.2 Event model

- Every event has:
  - `event_type` (from config)
  - `timepoint_ms` (anchor on `ms_axis`)
  - `range_ms` default **±100 ms** → `start_ms = timepoint_ms - 100`, `end_ms = timepoint_ms + 100`
  - User can edit range (spinboxes or duration field) before/after marking
- UI actions:
  - **Mark event** at current `timepoint_ms` (uses selected type + current range)
  - Edit selected event in list
  - Delete event
  - Click event row → seek to `timepoint_ms`

### 2.3 Per-block output file

Save under **output folder**, reloadable for refinement:

```json
{
  "schema_version": 1,
  "animal_call": "PV_126",
  "experiment_date": "2024_01_15",
  "block_num": "007",
  "block_path": "Z:/.../block_007",
  "created_at": "2026-05-22T14:30:00",
  "modified_at": "2026-05-22T16:45:00",
  "sync_source": "final_sync_df.csv",
  "sample_rate_hz": 30000,
  "ms_axis_range": [0.0, 123456.7],
  "config_snapshot": { "event_types": ["saccade", "blink", "noise", "pupil event"] },
  "events": [
    {
      "id": "uuid-or-increment",
      "event_type": "saccade",
      "timepoint_ms": 45230.5,
      "start_ms": 45130.5,
      "end_ms": 45330.5,
      "range_half_width_ms": 100,
      "row_index": 2718,
      "arena_frame": 120,
      "l_eye_frame": 118,
      "r_eye_frame": 119,
      "note": ""
    }
  ]
}
```

Filename convention: `{animal_call}_{experiment_date}_block_{block_num}_annotations.json` (omit date segment if `experiment_date` is null).

### 2.4 Open Ephys traces

- Use `BlockSync` → `block.oe_rec` (`OERecording` from `src/eye_tracking_system_tools/preprocessing/OERecording.py`).
- **Stream dropdown** must list available sources:
  - **Headstage / neural**: `oe_rec.channel_files` + `channelNumbers` → `get_data(channels, start_time_ms, window_ms, ...)`
  - **Analog ADC**: `oe_rec.analog_files` + `analogChannelNumbers` → `get_analog_data(...)`
  - **Accelerometer AUX**: `oe_rec.accel_files` → `get_accel_data(...)` (verify method name in `OERecording.py`)
- Label dropdown entries clearly, e.g. `HS ch12`, `ADC ch3`, `AUX accel_x`.
- **Time alignment**: `ms_axis` from `final_sync_df` uses the same OE sample clock as `Arena_TTL`. Pass `start_time_ms` to `get_data` as **absolute ms within zeroed OE recording** (same convention as `utility_functions.py` saccade/LFP snippets). Validate on one known block: scrub to a TTL feature and confirm trace window matches video.
- **Performance**: user-set **downsample factor** (integer ≥1). For trace view:
  - Precompute or cache a **decimated overview** of the selected stream across `[ms_min, ms_max]` for fast scrubbing.
  - Optional: higher-resolution lazy load in a ±window around playhead when paused.
- Missing / out-of-range data: do not plot (gap or flat “no data” region), no crash.

### 2.5 Videos

- Discover paths via `BlockSync` (`le_videos`, `re_videos`, `arena_videos` after `handle_arena_files` / existing init).
- **Arena**: dropdown of all detected arena MP4s; show **one** selected stream (not a grid of all).
- **Eyes**: L and R panels always visible.
- **Raw vs annotated** toggle per eye:
  - Raw: frame only; optional **horizontal flip of displayed frame only** (like `data_verification_utils` display path — `cv2.flip(frame, 1)`), **do not** mutate ellipse CSVs.
  - Annotated: overlay ellipse from `left_eye_data.csv` / `right_eye_data.csv` or `le_df` / `re_df` if present, keyed by `L_eye_frame` / `R_eye_frame` at current row. If overlay data missing, show raw + status text.
- Missing frame index or unreadable frame: black panel with centered **"missing frame"** text.

### 2.6 Playback

- Default **60 Hz** effective stepping along timeline rows (or time-based advance matching `ms_axis` deltas).
- User **speed multiplier** (e.g. 0.25×–4×):
  - **Slower than 1×**: must show every timeline step (no skipped rows).
  - **Faster than 1×**: may drop frames / skip rows for responsiveness.
- Transport: play/pause, slider, step back/forward (configurable step in rows), display `ms_axis`, row index, frame IDs.
- Adapt timer-based play from `GUI_example.py` `_on_timer` (wall-clock + `searchsorted` on `ms_axis`).

### 2.7 Timeline scope

- **All rows** in `final_sync_df` define the timeline (no dropping rows with NaN video frames).
- `n = len(final_sync_df)`; `ms_axis` length `n`.

---

## 3. Architecture

```
src/eye_tracking_system_tools/annotation/
  __init__.py
  block_annotator/
    __main__.py          # python -m eye_tracking_system_tools.annotation.block_annotator
    app.py               # QApplication, main window, menus
    models.py            # BlockSession, AnnotationEvent, Config
    config_io.py         # YAML load/save, template creation
    block_loader.py      # BlockSync wrapper, final_sync_df, ms_axis, video paths
    playback_controller.py
    video_widget.py      # QLabel + cv2 capture + cache + flip + ellipse overlay
    trace_widget.py      # pyqtgraph multi-channel / single stream view
    oe_streams.py        # enumerate streams, fetch decimated trace
    annotation_panel.py  # event list, mark/edit/delete, range controls
    persistence.py       # JSON save/load per block
```

### 3.1 Block loading flow

1. User selects block folder OR parent folder (if parent, show block picker when multiple `analysis/final_sync_df.csv` found).
2. Instantiate `BlockSync` minimally:
   - Parse `animal_call`, `experiment_date`, `block_num` from path if possible, else prompt user for missing metadata fields once.
   - Set `block_path`, call `load_final_sync_df(block)` from `block_sync_core`.
   - Resolve `sample_rate` (`block.sample_rate` or OE metadata).
   - Build `ms_axis` array.
   - Open `OERecording(block.oe_path)` if `oe_files` exist.
3. Discover videos; populate arena dropdown.

### 3.2 Dependencies to add

In `pyproject.toml` optional extra or main deps:

```toml
"PyQt6>=6.6",
"pyqtgraph>=0.13",
"PyYAML>=6.0",
```

Keep `opencv-python` (already present). Do **not** require VisPy/imageio unless arena decode via imageio is preferred; **prefer cv2.VideoCapture** for consistency with `BlockSync` / verification tools.

---

## 4. Implementation phases (agent loop)

Each phase ends with **acceptance checks**; agent must not proceed until checks pass.

### Phase 0 — Scaffold & launch

- [ ] Package layout + `python -m eye_tracking_system_tools.annotation.block_annotator` opens empty main window.
- [ ] Startup dialog: pick output folder, optional config, pick block folder.
- [ ] Auto-create `annotator_config.yaml` template if missing.

**Acceptance:** App launches on Windows; config template written; no import errors.

### Phase 1 — Block session & timeline

- [ ] Load `final_sync_df.csv`; compute `ms_axis`; expose `n`, `ms_axis`, frame columns.
- [ ] Status bar shows animal/block/date/ms range.
- [ ] Horizontal slider 0…n-1; labels update on scrub.

**Acceptance:** Load a real block from lab path; slider moves; `ms_axis` monotonic.

### Phase 2 — Video panel

- [ ] Arena dropdown + L/R videos; frame lookup from `Arena_frame`, `L_eye_frame`, `R_eye_frame`.
- [ ] LRU cache (32 frames per slot).
- [ ] Missing frame placeholder.
- [ ] Raw flip toggle (display only).
- [ ] Annotated overlay when CSVs exist.

**Acceptance:** Scrubbing updates all three panels in sync; missing frames show placeholder; flip does not alter saved ellipse files.

### Phase 3 — Playback

- [ ] Play/pause, step, speed multiplier, 60 Hz default.
- [ ] Slow-mo does not skip indices; fast-forward may skip.
- [ ] Keyboard: Space, arrows.

**Acceptance:** 30 s playback stays synchronized across videos; slow 0.5× hits every row.

### Phase 4 — OE trace panel

- [ ] `oe_streams.py` lists all stream/channel options from `OERecording`.
- [ ] Downsample factor control; decimated trace drawn under full timeline.
- [ ] Playhead cursor synced to `ms_axis[i]`.
- [ ] Alignment spot-check documented in code comments + manual test note.

**Acceptance:** Selecting a channel shows plausible signal; playhead tracks scrub; downsample=100 keeps scrub smooth on long blocks.

### Phase 5 — Annotations

- [ ] Event type dropdown from config; range default ±100 ms editable.
- [ ] Mark / edit / delete / list / seek on click.
- [ ] Save/load JSON per block in output folder.
- [ ] `modified_at` updated on save; reload merges or replaces (document: **replace all events** on save unless implementing merge-by-id in v1).

**Acceptance:** Save → close app → reload block → events restored; metadata fields populated.

### Phase 6 — Polish & docs

- [ ] README section: install, launch, config format, output JSON schema.
- [ ] Error dialogs: missing `final_sync_df`, no videos, no OE folder.
- [ ] Minimal smoke test script or `pytest` test for `config_io` + `ms_axis` computation (no GUI headless required for CI).

**Acceptance:** Fresh user can follow README on one example block.

---

## 5. Code reuse map

| Source | Reuse |
|--------|--------|
| `GUI_example.py` | `_video_frame_index_from_gui_index`, cache pattern, `_on_timer` time-based play, transport layout, HUD fields |
| `block_sync_core.load_final_sync_df` | Load sync CSV |
| `BlockSync_class.BlockSync` | Paths, sample_rate, videos, `oe_rec` |
| `OERecording.py` | `get_data`, `get_analog_data`, accel accessor, file lists |
| `data_verification_utils.py` | Display-only horizontal flip; ellipse draw logic from `interactive_ellipse_corrector` (read frame, lookup row by `eye_frame`) |

| Source | Do not port |
|--------|-------------|
| `GUI_example.py` VisPy scene, OBJ, 3D export, eye vectors, BNO, scene settings |

---

## 6. Known risks & mitigations

| Risk | Mitigation |
|------|------------|
| `ms_axis` vs `OERecording.allTimeStamps` offset | Unit test: compare `ms_axis[i]` to `get_data` window at same scrub point; document offset if Arena_TTL is block-relative |
| Full-rate `get_data` on every scrub | Decimated cache + downsample factor; only refresh hi-res when paused |
| Block path → `BlockSync` constructor needs animal/date | Infer from path regex; fallback dialog |
| PyQt6 not in current deps | Add to pyproject; document `pip install -e ".[annotator]"` |
| Large videos | cv2 seek; cache; consider read-ahead thread in v2 |

---

## 7. Agent prompt (copy below into a new Cursor Agent)

```
You are implementing the PETS Block Annotator GUI. Read and follow BLOCK_ANNOTATOR_IMPLEMENTATION_PLAN.md in the repo root completely.

Goal: Standalone PyQt6 app for synchronized review of arena + eye videos, Open Ephys traces, and user-defined event annotations on blocks with analysis/final_sync_df.csv.

Constraints:
- PyQt6 + OpenCV + pyqtgraph (add dependencies to pyproject.toml).
- Do NOT port 3D/VisPy from GUI_example.py; only reuse playback/video sync patterns.
- Master timeline = all rows of final_sync_df; ms_axis = Arena_TTL/(sample_rate/1000).
- User provides output folder at startup; auto-create annotator_config.yaml with event types: saccade, blink, noise, pupil event if missing.
- Events: timepoint_ms + default ±100ms range (user-editable); save per-block JSON in output folder with animal/block/date/timestamps and full event list.
- OE traces: dropdown of streams from OERecording (HS/ADC/AUX); align get_data windows to ms_axis; user downsample factor for performance.
- Arena: one video selected from dropdown; eyes: raw vs annotated toggle; raw display-only horizontal flip (never flip ellipse data).
- Missing video frames: black + "missing frame"; missing trace: no plot.
- Playback: 60Hz default, speed multiplier; no frame skip when speed < 1; may skip when speed > 1.
- Parallel tool — clean module under src/eye_tracking_system_tools/annotation/block_annotator/

Work in phases 0–6 from the plan. After each phase, run the acceptance checks listed there and fix failures before continuing.

Testing you must run yourself:
1. pip install -e . with new deps
2. python -m eye_tracking_system_tools.annotation.block_annotator
3. Load a real block (user will provide path if needed; use repo docs or ask)
4. Verify scrub, play, trace playhead, mark event, save, reload

Deliverables: working module, pyproject deps, README section, no unrelated refactors.

When uncertain about ms_axis ↔ OE alignment, read OERecording.py and utility_functions.py usage of get_data, then add a short comment in oe_streams.py explaining the convention you used.
```

---

## 8. How to run the agent and verify the GUI

### 8.1 One-shot implementation agent

1. Open Cursor Agent in `D:\Python_projects\PETS`.
2. Paste **Section 7** prompt.
3. Attach files: `BLOCK_ANNOTATOR_IMPLEMENTATION_PLAN.md`, `GUI_example.py`, `OERecording.py`, `block_sync_core.py` (load_final_sync_df), `data_verification_utils.py`.
4. Allow network for `pip install` when the agent adds dependencies.

### 8.2 Iterative convergence loop (recommended)

Use a **two-agent pattern**:

| Role | Task |
|------|------|
| **Builder agent** | Implements phases 0–6; fixes compile errors |
| **Reviewer agent** | Runs app + checklist; files concise failure report |

**Reviewer checklist (manual or agent):**

1. Startup: output folder + config template created.
2. Load block with `final_sync_df.csv` — no crash.
3. Scrub full slider — `ms_axis` increases; frame labels change.
4. All three videos stay in sync at same scrub index.
5. Toggle arena source in dropdown — video changes.
6. Raw flip on eye — image mirrors; re-open annotated — ellipses still correct coords.
7. Select OE stream, downsample=50 — scrub remains responsive.
8. Mark two events with different types/ranges — appear in list.
9. Save JSON — file exists in output folder with metadata.
10. Restart app, reload — events identical.
11. Play at 0.5× — no row skips; play at 2× — may skip.

**Cursor `/loop` skill** (local): after MVP exists, run e.g.

```
/loop 10m Run BLOCK_ANNOTATOR reviewer checklist: launch annotator, report pass/fail per item, fix builder issues if any fail.
```

Requires a **local** agent session (loop skill disabled in cloud).

### 8.3 Skills & tools

| Need | Tool / skill |
|------|----------------|
| Phased implementation | Default **Agent** mode with Section 7 prompt |
| Recurring QA | **`loop` skill** (`/loop 10m ...`) |
| PR hygiene later | `babysit` skill (optional, not needed now) |
| GUI automation | **No browser MCP** — this is desktop Qt; reviewer must run Python locally |
| Shell | Agent **Shell** for `pip install`, `python -m ...` |

### 8.4 Launch command (after implementation)

```powershell
cd D:\Python_projects\PETS
pip install -e .
python -m eye_tracking_system_tools.annotation.block_annotator
```

---

## 9. Approval record

- Spec converged from user Q&A on 2026-05-22.
- Approved implicitly by user request to generate this plan and agent prompt.
