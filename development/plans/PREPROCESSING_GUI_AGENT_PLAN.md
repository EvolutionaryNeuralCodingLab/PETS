# Preprocessing GUI — Implementation Plan & Agent Prompt

This document is the single source of truth for building the **Preprocessing GUI**: a PyQt6 application that packages every interactive step from the five preprocessing notebooks into one tool, taking a user from an organized-but-empty `block_xxx/` folder all the way to verified, synchronized, Kerr-converted, behaviorally-annotated eye data.

It is written as a direct prompt for the implementing agent. Read sections 0–2 first, then follow sections 4–6 in order. Sections 7–8 are the user-facing acceptance gates.

---

## 0. Mission summary

Build `eye_tracking_system_tools.annotation.preprocessing_gui` — a single PyQt6 app, launched via `python -m eye_tracking_system_tools.annotation.preprocessing_gui`, that exposes **all** interactive functionality currently scattered across:

1. `src/eye_tracking_system_tools/preprocessing/block_synchronization.ipynb`
2. `src/eye_tracking_system_tools/preprocessing/data_verification.ipynb`
3. `src/eye_tracking_system_tools/preprocessing/kerr_degree_conversion.ipynb`
4. `src/eye_tracking_system_tools/preprocessing/add_accelerometer_state_annotations.ipynb`
5. `src/eye_tracking_system_tools/preprocessing/sync_free_eye_ellipse_pipeline.ipynb`

The GUI **calls the existing `BlockSync` / helper functions verbatim** wherever possible. It does not reimplement synchronization logic; it provides a UI shell, parameter capture, interactive plots, and progress orchestration around them. The notebooks remain the authoritative source of the underlying algorithms.

`manual_outlier_annotation.ipynb` is **explicitly out of scope** for v1.

---

## 1. Locked design decisions (from user interview, 2026‑05‑26)

| # | Decision | Locked answer |
|---|----------|---------------|
| 1 | GUI framework / env | **PyQt6**, runs in **`eye_repo_win`** (Windows) or **`eye_repo_linux`**. `cv2` is used **only headlessly** (`cv2.VideoCapture`, `cv2.warpAffine`, etc.). **No** `cv2.imshow` / `cv2.selectROI` / `cv2.namedWindow` anywhere in the GUI code path — they conflict with PyQt6 on Windows (see `SETUP_INSTRUCTIONS.md`). |
| 2 | Overall layout | **Tabbed dashboard**: one tab per stage, free navigation, status icons per tab show whether the corresponding outputs exist on disk for the current block. |
| 3 | Block scope | **Single block at a time** is the primary interactive mode. The long automatic steps (`get_eye_brightness_vectors`, `read_dlc_data`, `get_jitter_reports`, `calculate_kerr_angles`) also expose a **"Run for all selected blocks"** batch button. |
| 4 | Bokeh plots | **Hybrid.** The slider-based shift-correction plot stays as the existing Bokeh-in-browser figure (its JS slider semantics are exactly what we want and re-implementing them is error-prone). The simpler sanity / jitter plots are re-implemented natively with **pyqtgraph** inside the GUI. |
| 5 | Eye+ellipse verifier | **Native Qt widget.** Replace the `cv2.imshow` "Controls" / "Frame" windows from `data_verification_utils.interactive_ellipse_corrector` with a `QGraphicsView`-based widget supporting play/pause, scrubber, ±1‑minute skip, transformation buttons (X‑flip, Phi+90, FlipX‑only, Flip Dot), and click‑to‑pick Kerr reference. The numeric helpers (`horizontal_flip_eye_data`, `rotate_phi_only`, `flip_x_only`) are imported and reused unchanged. |
| 6 | Manual TTL fallback | **Either approach acceptable.** Recommended: build a Qt dialog that produces `manual_line_map` + `arena_window` dicts and pass them to `parse_open_ephys_events` (the method already accepts those overrides via its lower-level path — confirm by inspecting `_manual_ttl_map_and_window`). Re-use `_summarize_ttl_lines_from_events_csv` and `_plot_ttl_raster` for the dialog content. No `input()` prompts in the GUI. |
| 7 | Brightness ROI | **Auto first, Qt rubber-band fallback.** Try `get_roi_auto_brightest_2x2`; on failure open a native Qt picker (still frame in a `QGraphicsView`, rubber-band selection). Pass the resulting ROI to `produce_frame_val_list_with_roi`. |
| 8 | Sync-free pipeline | **Include as a separate tab** in the same GUI, sharing the Kerr-ref picker and ellipse verifier widget. |
| 9 | Manual outlier notebook | **Out of scope for v1.** |
| 10 | Self-check during build | **pytest + headless validation script.** Add `tests/test_preprocessing_gui_*.py` for pure-Python logic (no Qt event loop required), AND a `scripts/validate_preprocessing_gui_load.py` that opens the GUI headlessly on the sample block and instantiates every tab without crashing (mirrors `scripts/validate_block_annotator_load.py`). |
| 11 | Human end-to-end check | **`PREPROCESSING_GUI_ACCEPTANCE.md`** checklist with numbered manual tests per UI feature, plus expected on-disk outputs to diff against notebook-produced reference outputs. |
| 12 | Sample data | `D:\sample_data_for_eye_repo\PV_106\2025_09_04\block_015` is the **reference sample block** for self-checks and acceptance. |

---

## 2. Repository context the agent must know

### 2.1 Where similar GUIs already live (study these first)

- `src/eye_tracking_system_tools/annotation/block_annotator/` — best precedent. Mirror its file layout, naming, and entry point (`app.py` with `main()`, `__main__.py` thin launcher, `models.py`, `persistence.py`, `config_io.py`, `video_widget.py`, `trace_widget.py`, `playback_controller.py`).
- `src/eye_tracking_system_tools/annotation/event_explorer/` — second precedent, especially for catalog/session-style state.
- `scripts/validate_block_annotator_load.py` — exact pattern for the headless validation script.
- `tests/test_block_annotator.py`, `tests/test_event_explorer.py` — pytest patterns.
- Existing planning docs to imitate in tone: `development/plans/EVENT_EXPLORER_IMPLEMENTATION_PLAN.md`.

### 2.2 Key APIs the GUI will wrap (do not rewrite)

| Capability | Symbol | File |
|------------|--------|------|
| Block discovery | `block_generator(block_numbers, experiment_path, animal, bad_blocks)` | `preprocessing/utility_functions.py` |
| Construct block | `BlockSync(animal_call, experiment_date, block_num, path_to_animal_folder, channeldict=None)` | `preprocessing/BlockSync_class.py` |
| Stage 1 — videos & OE | `BlockSync.handle_eye_videos()`, `BlockSync.parse_open_ephys_events(...)`, `BlockSync.handle_arena_files()`, `BlockSync.get_eye_brightness_vectors(use_auto_roi=True, create_if_missing=True)` | `BlockSync_class.py` |
| Stage 1 — arena grid | `build_arena_grid_df(block, target_fps=60.0, arena_fps_tol_hz=5.0)` | currently defined inline in `block_synchronization.ipynb` Cell 1 — **promote to a new module** `preprocessing/block_sync_core.py` extensions or a new `preprocessing/notebook_helpers.py` (see §4.3). |
| Stage 1 — simple sync | `simple_sync_build(block, export=True)`, `plot_simple_sync_bokeh(block, dfL, dfR, show_led=True)`, `shift_eye_df_by_index(df, shift)`, `insert_dup_by_pos`, `insert_dup_by_oe_sample` | same — promote out of notebook |
| Stage 1 — final merge & verify | `build_final_sync_df_merge_nearest(...)`, `sanity_plot_final_df(...)`, `verify_final_df_against_sources(...)`, `export_final_sync_df(...)`, `load_final_sync_df(...)` | `preprocessing/block_sync_core.py` (already has `load_final_sync_df`; add the rest) |
| Stage 1 — DLC + jitter | `BlockSync.read_dlc_data(threshold_to_use=0.95, export=True, overwrite=False)`, `BlockSync.get_jitter_reports(...)`, `BlockSync.correct_jitter()`, `BlockSync.find_led_blink_frames(plot=False)`, `BlockSync.remove_led_blinks_from_eye_df(export=True)`, `find_jittery_frames(block, eye, max_distance, diff_threshold, gap_to_bridge)` | `BlockSync_class.py` + notebook helper |
| Stage 1 — finalize | `BlockSync.remove_eye_datapoints_based_on_video_frames(eye, indices_to_nan, export)`, `BlockSync.create_eye_data()`, `export_eye_data_2d(block)` | `BlockSync_class.py` + notebook helper |
| Stage 2 — verification helpers | `interactive_ellipse_corrector(df, video_path, eye, ref_point_xy, block=None, on_save=None)` (will be **replaced** by the new Qt widget but its transformation helpers stay: `horizontal_flip_eye_data`, `rotate_phi_only`, `flip_x_only`), `export_corrected_eye_data(block)`, `export_current_kerr_refs(block)`, `load_self_kerr_refs(block)` | `preprocessing/data_verification_utils.py` |
| Stage 3 — Kerr | `BlockSync.calculate_kerr_angles(name_tag)`, `append_angle_data(eye_df, new_df)` (helper from `kerr_degree_conversion.ipynb`) | `BlockSync_class.py` + notebook helper |
| Stage 4 — accelerometer | `BlockSync.block_get_lizard_movement()` → sets `block.liz_mov_df`; helpers `rolling_window_analysis(df, window_size, step_size)`, `create_behavior_df(df)` currently in the notebook | promote helpers out of notebook |
| Sync-free | `run_syncfree_ellipses_for_eye`, `append_kerr_columns_syncfree`, `map_syncfree_degrees_to_final_sync`, `default_syncfree_paths`, `write_meta_json`, `write_self_kerr_refs_csv` | `preprocessing/sync_free_eye_io.py` |

### 2.3 Block folder layout assumed by `BlockSync`

```
path_to_animal_folder/
  <animal>/
    <yyyy_mm_dd>/          (or no date level if experiment_date=None)
      block_<NNN>/
        arena_videos/      (flat or under /videos/)
        eye_videos/LE/<sub>/<*.mp4 + DLC*.csv + *_timestamps.csv>
        eye_videos/RE/<sub>/<*.mp4 + DLC*.csv + *_timestamps.csv>
        oe_files/<exp_dt>/Record Node <N>/{settings.xml, *.continuous, *.events, ...}
        analysis/          (initially empty; this GUI writes here)
```

Reference sample block: `D:\sample_data_for_eye_repo\PV_106\2025_09_04\block_015`.

### 2.4 Outputs the GUI must produce (acceptance criteria)

Inside `block/analysis/`:

| File | Produced by which tab |
|------|-----------------------|
| `arena_brightness.csv` | Sync (`handle_arena_files`) |
| `eye_brightness_values_dict.pkl` | Sync (`get_eye_brightness_vectors`) |
| `parsed_events.csv` (under `oe_files/<exp_dt>/`) | Sync (`parse_open_ephys_events`) |
| `eye_left_simple_sync.csv` / `eye_right_simple_sync.csv` | Sync (`simple_sync_build`) |
| `blocksync_df.csv` and `final_sync_df.csv` | Sync (`export_final_sync_df`) |
| `le_df.csv`, `re_df.csv` | Sync (jitter / blink stage) |
| `left_eye_data.csv`, `right_eye_data.csv` | Sync final export |
| `self_kerr_refs.csv` | Verify (`export_current_kerr_refs`) |
| `left_kerr_angle_<tag>.csv`, `right_kerr_angle_<tag>.csv` | Kerr (`calculate_kerr_angles`) |
| `left_eye_data_<tag>.csv`, `right_eye_data_<tag>.csv` | Kerr (export with angles) |
| `block_<num>_behavior_state.csv` | Behavior |
| `{left,right}_eye_degrees_from_syncfree_<tag>.csv` | Sync-free tab (optional) |

Inside each eye video folder (sync-free only):
- `{side}_syncfree_<tag>_ellipses.csv`, `*_meta.json`, `*_verified.csv`, `*_kerr_refs.csv`, `*_kerr_angles.csv`, `*_degrees.csv`

---

## 3. Functional inventory — notebook → GUI mapping

Each row is a unit of interactive work the GUI must expose. "User input" means a place where a notebook cell currently asks for human judgement.

### Stage 1 — Synchronization (block_synchronization.ipynb)

| # | Notebook cell intent | GUI control | User input | Backend call |
|---|----------------------|-------------|------------|--------------|
| 1.1 | Pick `experiment_path`, `animal`, `block_numbers`, `bad_blocks`, optional `channeldict` | "Block setup" form at top of Sync tab; folder picker for `experiment_path`; multi-select for block numbers; optional channeldict editor | Yes | `uf.block_generator(...)`, then attach `channeldict` |
| 1.2 | `handle_eye_videos()`, `handle_arena_files()` | "Prepare data" button | No (auto) | Method calls |
| 1.3 | `parse_open_ephys_events()` — auto path | "Parse OE events" button | Mostly no | `block.parse_open_ephys_events(overwrite=…)` |
| 1.4 | Manual fallback: TTL line mapping + window selection | "Manual TTL mapping…" dialog (only shown if auto fails or user clicks "Override"): line summary table, raster preview (pyqtgraph), role→line mapping form, window-mode radio (i/s/a) | Yes | Build `manual_line_map` + `arena_window` and re-call `parse_open_ephys_events` (re-use `_manual_ttl_map_and_window` internals; do not duplicate logic) |
| 1.5 | `get_eye_brightness_vectors(use_auto_roi=True)` | "Extract brightness" button | Only on auto-ROI failure | `block.get_eye_brightness_vectors(use_auto_roi=True, create_if_missing=False)`; on failure open Qt ROI picker per eye, then call `produce_frame_val_list_with_roi` directly |
| 1.6 | `build_arena_grid_df(target_fps=60, arena_fps_tol_hz=5)` | "Build arena grid" with spinboxes for the two parameters | Sometimes (tuning) | `build_arena_grid_df(...)` |
| 1.7 | `simple_sync_build(block, export=True)` | "Build simple sync" button; show resulting `dfL`/`dfR` summary (rows, median fps, CoV) | No | `simple_sync_build(block, export=True)` |
| 1.8 | Interactive Bokeh shift-correction (`plot_simple_sync_bokeh`) | "Open shift plot in browser" button + two spinboxes "Left shift (ticks)" / "Right shift (ticks)" (defaults 0). After typing values: "Apply shifts" button | Yes | Open Bokeh in browser via the existing `plot_simple_sync_bokeh(block, dfL, dfR, show_led=True)`; user notes the shift values and types them back in. Spinbox step = 1; show "≈ X ms" next to each spinbox (computed from `describe_eye_tick(df)`). |
| 1.9 | Optional: dropped-frame correction (`insert_dup_by_pos` / `insert_dup_by_oe_sample`) | Collapsible "Frame insertion (advanced)" panel: per-eye add-row table (position OR oe_sample, duplicate mode), "Apply insertions" button | Yes (advanced) | `insert_dup_by_pos` / `insert_dup_by_oe_sample` |
| 1.10 | `build_final_sync_df_merge_nearest(...)` | "Build final sync df" with spinbox `tol_frac` (default 0.9) | Sometimes | The function |
| 1.11 | `sanity_plot_final_df` + `verify_final_df_against_sources` | "Verify" button. Show stats in a `QLabel`/`QPlainTextEdit` ("Left frame match: 99.998% …"). Plot in embedded **pyqtgraph** widget. Re-implement the brightness/LED overlay natively. | No | Both functions |
| 1.12 | `export_final_sync_df(block, final_df, overwrite=True)` | "Export final_sync_df.csv" button (greyed out until 1.11 passes) | No | The function |
| 1.13 | `read_dlc_data(overwrite=False, export=True)` | "Read DLC + fit ellipses" button + spinbox for `threshold_to_use` (default 0.95) | No (param) | `block.read_dlc_data(threshold_to_use=..., overwrite=..., export=True)` |
| 1.14 | `get_jitter_reports(export=True, overwrite=False, remove_led_blinks=False, sort_on_loading=True)` | "Compute jitter report" button (long-running; show progress) | No | Method |
| 1.15 | `correct_jitter()` + `find_led_blink_frames(plot=False)` + `remove_led_blinks_from_eye_df(export=True)` | "Correct jitter & remove LED blinks" button | No | Three calls in sequence |
| 1.16 | `find_jittery_frames` + `uf.bokeh_plotter` for review | Per-eye panel: three spinboxes (`max_distance` default 60, `diff_threshold` default 5, `gap_to_bridge` default 24); "Preview outliers" button → embedded pyqtgraph plot of `top_correlation_dist` with marked peaks; "Apply removal" button | Yes (tuning) | `find_jittery_frames`, then `block.remove_eye_datapoints_based_on_video_frames(eye, indices_to_nan=vid_inds)` |
| 1.17 | `create_eye_data()` + `export_eye_data_2d(block)` | "Finalize & export eye data" button | No | Two calls |
| 1.18 | Multi-block batch | Each long-running step (1.5, 1.13, 1.14, 1.15) has a "Run for all selected blocks" checkbox/button | Yes | Loop the call over the user's selected blocks |

### Stage 2 — Data Verification (data_verification.ipynb)

| # | Notebook cell intent | GUI control |
|---|----------------------|-------------|
| 2.1 | Load CSVs | Auto on tab activation (`load_eye_data(block)`) |
| 2.2 | Run interactive ellipse corrector for each eye | **New native Qt widget** `EllipseVerifierWidget` (see §4.4): video display via `cv2.VideoCapture` → `numpy_rgb_to_qpixmap` (re-use the helper from `block_annotator/video_widget.py`), QGraphicsView for click-to-pick-ref, buttons: Play, Pause, X-flip, Phi+90, FlipX-only, Flip Dot, Bwd (1 min), Fwd (1 min), Save, Quit. Show frame slider + frame counter. Display y-flipped to keep the "y-positive = up" convention. Live ellipse overlay using the current `df_current` row. |
| 2.3 | Export corrected eye data | "Save & export" button → calls `export_corrected_eye_data(block)` |
| 2.4 | Save Kerr ref point | The verifier's Save also writes `self_kerr_refs.csv` via `export_current_kerr_refs(block)` |

### Stage 3 — Kerr Degree Conversion (kerr_degree_conversion.ipynb)

| # | Intent | GUI control |
|---|--------|-------------|
| 3.1 | Load eye data | Auto on tab activation; require `load_self_kerr_refs(block)` succeeds, else show "Pick Kerr refs in Verify tab first" message |
| 3.2 | `calculate_kerr_angles(name_tag)` | Text field for `name_tag` (default `raw_verified`); "Calculate" button |
| 3.3 | Append + export `*_eye_data_<tag>.csv` | "Export merged" button using `append_angle_data` helper |
| 3.4 | Multi-block | "Run for all selected blocks" checkbox |

### Stage 4 — Accelerometer State Annotations (add_accelerometer_state_annotations.ipynb)

| # | Intent | GUI control |
|---|--------|-------------|
| 4.1 | `block_get_lizard_movement()` | "Load lizMov.mat" button; handle FileNotFoundError gracefully with a clear "Run the MATLAB getLizMovement function first" message |
| 4.2 | Rolling window | Two spinboxes: `window_size_ms` (default 10000), `step_size_ms` (default 1000); "Compute rolling avg" button |
| 4.3 | Threshold selection | Embedded **pyqtgraph** plot of `average_movAll` vs `window_start` with a draggable horizontal threshold line (use `pg.InfiniteLine(angle=0, movable=True)`); current threshold shown in a spinbox bound to the line; coloured shading or two-tone bars showing active/quiet segments live |
| 4.4 | Export | "Export behavior state" button writes `block_<num>_behavior_state.csv` with columns `start_time, end_time, annotation` (re-use `create_behavior_df(df)` helper) |

### Tab 5 — Sync-free pipeline (sync_free_eye_ellipse_pipeline.ipynb)

| # | Intent | GUI control |
|---|--------|-------------|
| 5.1 | Parameters | Text fields: `artifact_tag` (default `v1`), `uncertainty_thr` (default 0.95); checkboxes for `RUN_ELLIPSES`, `RUN_VERIFY`, `RUN_KERR`, `RUN_MAP` |
| 5.2 | Run ellipses | "Run ellipses" button → `run_syncfree_ellipses_for_eye` per eye |
| 5.3 | Verify | Reuses the **same** `EllipseVerifierWidget` from Stage 2 (parametrized with a different on-save callback that writes the sync-free `*_verified.csv` + sidecar Kerr refs and merges them via `_merge_self_kerr_refs`) |
| 5.4 | Kerr | "Append Kerr" → `append_kerr_columns_syncfree` per eye |
| 5.5 | Map to final_sync | "Map to final_sync_df" → `map_syncfree_degrees_to_final_sync`. Greyed out and show staleness warning if `final_sync_df.csv` mtime > current mapped file mtime |

---

## 4. Architecture & module layout

### 4.1 Package location

```
src/eye_tracking_system_tools/annotation/preprocessing_gui/
  __init__.py
  __main__.py                  # thin launcher, like block_annotator/__main__.py
  app.py                       # MainWindow + main()
  models.py                    # GuiState dataclass, per-stage status enums
  config_io.py                 # YAML config (defaults + last-used paths)
  block_picker.py              # animal/date/block multi-select widget + channeldict editor
  status_bus.py                # Qt signal hub: which outputs exist on disk → tab status icons
  bokeh_launcher.py            # opens bokeh figures in default browser, returns tmp html path
  qt_roi_picker.py             # Qt rubber-band ROI picker (still frame)
  manual_ttl_dialog.py         # Qt dialog for manual TTL mapping + window selection
  ellipse_verifier.py          # EllipseVerifierWidget (replaces cv2.imshow)
  pyqtgraph_helpers.py         # native versions of sanity_plot_final_df / bokeh_plotter
  tabs/
    __init__.py
    sync_tab.py                # Stage 1 (the big one)
    verify_tab.py              # Stage 2
    kerr_tab.py                # Stage 3
    behavior_tab.py            # Stage 4
    syncfree_tab.py            # Stage 5
```

### 4.2 Entry point

`python -m eye_tracking_system_tools.annotation.preprocessing_gui [--experiment-path PATH] [--animal STR] [--block STR] [--dialog]`

- With no args → show a startup dialog (mirror `block_annotator.StartupDialog`) asking for experiment path, animal, and block(s).
- With args → skip the dialog and load the specified block(s) directly.
- `--dialog` → force the startup dialog even if env vars are set.
- Honour env vars `PETS_PREPROC_EXPERIMENT_PATH`, `PETS_PREPROC_ANIMAL`, `PETS_PREPROC_BLOCKS` for consistency with the annotator convention.

### 4.3 Refactor required in `preprocessing/`

The notebook helpers currently defined inside `block_synchronization.ipynb` Cell 1 must be **promoted to a real module** so the GUI can `import` them. Create:

```
src/eye_tracking_system_tools/preprocessing/notebook_helpers.py
```

…and move the following functions (verbatim, no logic changes — just import-friendly):

- `_normalize_to_seconds`, `_read_eye_internal_seconds`, `_locate_eye_timestamps_csv`, `_delta_analysis`, `_first_ttl_sample`, `_get_fs`, `_assert_strictly_increasing`, `_nearest_with_tol`, `_shift_eye_df_by_index`
- `build_eye_df_simple`, `simple_sync_build`, `describe_eye_tick`, `shift_eye_df_by_index`
- `build_arena_grid_df`, `ArenaGridInfo`, `_infer_ttl_fps`, `_build_arena_grid`
- `build_final_sync_df_merge_nearest`, `sanity_plot_final_df`, `verify_final_df_against_sources`, `export_final_sync_df`
- `plot_simple_sync_bokeh`, `hover_inspect_eyes_bokeh`
- `_normalize_insert_positions`, `insert_duplicate_frames_slide`, `insert_dup_by_pos`, `insert_dup_by_oe_sample`
- `find_jittery_frames`, `add_intermediate_elements`, `export_eye_data_2d`

`load_final_sync_df` already exists in `preprocessing/block_sync_core.py` — re-export it from `notebook_helpers` for one-stop importing.

For Stage 4: also promote `rolling_window_analysis` and `create_behavior_df` from `add_accelerometer_state_annotations.ipynb` into `notebook_helpers.py` (or a new `accelerometer_helpers.py` if the agent prefers a clean split — either is fine; document the choice).

**Crucial:** after the refactor, edit the notebooks themselves to `from eye_tracking_system_tools.preprocessing.notebook_helpers import ...` instead of defining inline. The notebooks must still run end-to-end with the same results (validated as part of self-check, see §6).

### 4.4 EllipseVerifierWidget — design notes

Mirror the API of the existing `data_verification_utils.interactive_ellipse_corrector` so callers can opt in to the Qt widget without re-engineering Stage 5:

```python
class EllipseVerifierWidget(QWidget):
    saved = pyqtSignal(object, object)  # (df_corrected, ref_xy_or_None)
    quit_requested = pyqtSignal()

    def __init__(self, df, video_path, eye, ref_point_xy=None, parent=None): ...
    def df(self) -> pd.DataFrame: ...
    def ref_xy(self) -> tuple[int, int] | None: ...
```

Behaviour:
- Top: `QGraphicsView` showing the current video frame, **vertically flipped for display** (preserve "y-positive = up" convention) with the ellipse drawn from the current row of `df_current` and a blue 5-px dot at `current_ref` (in raw coords).
- Click on the frame → set `current_ref = (x_raw, H - 1 - y_view)` (matches `on_mouse_frame` semantics in the cv2 version).
- Below: transport row (Play, Pause, Bwd 1 min, Fwd 1 min, frame slider, frame counter `i / N`).
- Below: transformation buttons (X‑flip, Phi+90, FlipX‑only, Flip Dot) calling `horizontal_flip_eye_data`, `rotate_phi_only`, `flip_x_only` from `data_verification_utils`.
- Bottom: Save, Quit. Save emits `saved` with current df and ref.
- Frame reading: re-use `VideoReader` and `numpy_rgb_to_qpixmap` from `annotation/block_annotator/video_widget.py` (do not duplicate). If the existing reader has a different API, wrap it.

### 4.5 Shift-correction interaction (Stage 1.8 detail)

To keep the existing Bokeh slider behaviour without re-implementing it:

1. User clicks "Open shift plot". GUI calls `plot_simple_sync_bokeh(block, dfL, dfR, show_led=True, to_browser=True)`; Bokeh opens a tab in the default browser with the existing JS slider.
2. The GUI shows a small modal-less dock: two spinboxes ("Left shift" / "Right shift", integer, step 1, with live "≈ X.XX ms" computed from `describe_eye_tick`), a memo line ("Use the browser plot to find shifts that align LED rising edges; type them here, then click Apply"), and an "Apply shifts" button.
3. "Apply shifts" calls `shift_eye_df_by_index(dfL, -L_shift)` / `shift_eye_df_by_index(dfR, -R_shift)` (note: the notebook documents that GUI inputs are **inverse** to plot slider values — preserve this convention and document it next to the spinboxes).
4. "Preview applied" reopens the Bokeh plot with the shifted dataframes so the user can confirm.

### 4.6 Status-bus convention

Each tab declares a `status_signature()` that returns a list of `Path`s whose existence (and optionally mtime ordering) implies that stage is "done" for the current block:

```python
def sync_status_signature(block):
    return [
        block.analysis_path / "final_sync_df.csv",
        block.analysis_path / "left_eye_data.csv",
        block.analysis_path / "right_eye_data.csv",
    ]
```

The status bus polls these on tab change (and on a `QFileSystemWatcher` event) and updates the tab icon (green = complete, yellow = partial, grey = not started, red = stale-sync detected). Stale sync = `final_sync_df.csv` mtime newer than any downstream `*_eye_data*.csv` / Kerr CSV — same rule as the Event Explorer.

---

## 5. Build phases (in order)

Each phase ends with a self-check step (§6). Do **not** start phase N+1 until phase N's self-check is green.

### Phase 0 — Skeleton & smoke

1. Create the package directories per §4.1.
2. Implement `app.py` with `MainWindow` containing one `QTabWidget` and five empty `QWidget` tabs labelled Sync / Verify / Kerr / Behavior / Sync-free.
3. Implement `__main__.py` and `app.main()` with `argparse` for `--experiment-path / --animal / --block / --dialog`.
4. Implement `StartupDialog` (mirror `block_annotator.StartupDialog` but for picking experiment path / animal / block).
5. Implement `block_picker.py` and wire it so the chosen block(s) populate a `GuiState` dataclass shared across tabs.
6. Implement `status_bus.py` (file-watcher only; tab icons can be a stub at this stage).
7. Refactor §4.3 — promote notebook helpers to `notebook_helpers.py`; edit notebooks to import from it; re-run all five notebooks end-to-end on the sample block and confirm outputs are identical.

**Self-check 0:** `scripts/validate_preprocessing_gui_load.py --experiment-path D:\sample_data_for_eye_repo --animal PV_106 --block 015` opens the GUI headlessly (offscreen QPA), instantiates all five tabs, and exits 0.

### Phase 1 — Sync tab, deterministic half (1.1 – 1.12)

1. Implement the Sync tab as a `QSplitter` (left = step list with status icons, right = active step panel).
2. Implement controls 1.1–1.7 and 1.10–1.12 (all the non-interactive ones).
3. Implement 1.8 (shift correction) via `bokeh_launcher.py` + spinboxes per §4.5.
4. Implement 1.9 (frame insertion) as a collapsible advanced panel.
5. Replace `sanity_plot_final_df` with a native pyqtgraph version in `pyqtgraph_helpers.py`; original function still callable for notebook users.
6. Wire `verify_final_df_against_sources` and display its return dict in a read-only text widget.

**Self-check 1:** new pytest in `tests/test_preprocessing_gui_sync.py`:
- `test_promoted_helpers_match_notebook` — calls `simple_sync_build`, `build_arena_grid_df`, `build_final_sync_df_merge_nearest`, `verify_final_df_against_sources` on the sample block and asserts the same column sets and row counts as the notebook reference.
- `test_shift_eye_df_by_index_inverse` — invariant: `shift_eye_df_by_index(shift_eye_df_by_index(df, n), -n)` reproduces the original ignoring edge NaNs.
- `test_status_signature_sync` — touching `final_sync_df.csv` flips the tab's status to green.

Update `scripts/validate_preprocessing_gui_load.py` to additionally instantiate Sync tab widgets without crashing.

### Phase 2 — Sync tab, ellipse + jitter half (1.13 – 1.18)

1. Wire `read_dlc_data`, `get_jitter_reports`, `correct_jitter`, `find_led_blink_frames`, `remove_led_blinks_from_eye_df` as buttons with `QThread` workers (these are long-running; the UI must stay responsive).
2. Implement the per-eye jitter review panel with embedded pyqtgraph (`top_correlation_dist` line + scattered peaks).
3. Implement `find_jittery_frames` parameter capture (spinboxes), preview, apply.
4. Implement `create_eye_data` + `export_eye_data_2d` finalization button.
5. Implement multi-block batch dispatch (sequential, with a global cancel button and per-block progress in a bottom log dock).

**Self-check 2:** pytest:
- `test_find_jittery_frames_matches_notebook` — same `max_distance=60, diff_threshold=5, gap_to_bridge=24` as the notebook gives the same `vid_inds_l / vid_inds_r` arrays on the sample block (after the deterministic upstream steps have been run).
- `test_status_signature_eye_data_csv` — green after `left_eye_data.csv` and `right_eye_data.csv` both exist.

### Phase 3 — Manual fallbacks (TTL, ROI)

1. Implement `manual_ttl_dialog.py`. Re-use `_summarize_ttl_lines_from_events_csv` and `_plot_ttl_raster` (you may need to refactor `_plot_ttl_raster` so its plot can be embedded; if too invasive, keep a "Show raster in browser" button inside the dialog). Output: `manual_line_map: dict[str, int]` + `arena_window: dict`. Call `parse_open_ephys_events` with these (the public method already supports the fallback path internally — confirm by reading lines 1387-1408 of `BlockSync_class.py`).
2. Implement `qt_roi_picker.py`: load first frame via `cv2.VideoCapture`, show in `QGraphicsView` with a `QRubberBand`; return ROI as `(x, y, w, h)` tuple compatible with `produce_frame_val_list_with_roi`.
3. Wire both into the Sync tab — they appear automatically when auto paths fail, or via "Override" buttons.

**Self-check 3:** pytest:
- `test_manual_ttl_dialog_payload_shape` — instantiate the dialog with a fixture events CSV, simulate selections programmatically, assert the produced `manual_line_map` + `arena_window` match a hand-crafted golden dict.
- `test_qt_roi_picker_returns_tuple` — instantiate with a 64x64 dummy frame, programmatically set the rubber-band geometry, assert the returned tuple equals the expected bounds.

### Phase 4 — Verify tab + EllipseVerifierWidget

1. Implement `EllipseVerifierWidget` per §4.4. Verify the y-flip + click-to-pick math matches the cv2 version exactly (use the cv2 version's `on_mouse_frame` line `y_raw = H - 1 - y` as the spec).
2. Wire the Verify tab: load eye data CSVs on tab activation, show two `EllipseVerifierWidget` instances side by side (one per eye), wire `saved` signals to call `export_corrected_eye_data(block)` + `export_current_kerr_refs(block)`.
3. The Verify tab must show a banner if the eye-data CSVs are stale (mtime older than `final_sync_df.csv`).

**Self-check 4:** pytest:
- `test_ellipse_verifier_yflip_click_math` — instantiate widget, programmatically simulate a click at view coords `(100, 50)` on a frame of height `H`; assert the stored ref is `(100, H-1-50)`.
- `test_ellipse_verifier_save_emits` — connect a slot, call `_on_save_clicked()`, assert it emitted `(df_current, ref_xy)`.
- `test_verify_tab_writes_self_kerr_refs` — drive the tab end-to-end on the sample block, assert `analysis/self_kerr_refs.csv` exists with all four `kerr_ref_*` columns populated.

### Phase 5 — Kerr tab

1. Implement Stage 3 UI per §3 of the inventory.
2. Implement multi-block batch.

**Self-check 5:** pytest:
- `test_kerr_tab_produces_csvs` — drive end-to-end on the sample block with `name_tag='gui_test'`, assert `left_kerr_angle_gui_test.csv`, `right_kerr_angle_gui_test.csv`, `left_eye_data_gui_test.csv`, `right_eye_data_gui_test.csv` all exist with `k_phi`, `k_theta` columns.

### Phase 6 — Behavior tab

1. Implement Stage 4 UI per §4 of the inventory. The interactive threshold widget is a pyqtgraph plot with a draggable `InfiniteLine`; bind it bidirectionally to a `QDoubleSpinBox`.
2. Handle the FileNotFoundError on `block_get_lizard_movement()` gracefully (clear instructional message).

**Self-check 6:** pytest:
- `test_rolling_window_analysis_helpers` — `rolling_window_analysis` and `create_behavior_df` produce expected schemas on a synthetic `liz_mov_df` fixture.
- `test_behavior_tab_writes_csv` — drive end-to-end on the sample block (if `lizMov.mat` is present; otherwise mark `xfail` with reason "no lizMov.mat in sample data").

### Phase 7 — Sync-free tab

1. Implement Stage 5 UI per §5 of the inventory.
2. Reuse `EllipseVerifierWidget` with an `on_save` callback that writes sync-free artifacts (see notebook Cell 4 for the callback shape).

**Self-check 7:** pytest:
- `test_syncfree_outputs_match_notebook` — run all four sub-stages on the sample block, assert that the produced files under the eye folder + the mapped CSV under `analysis/` have the same column sets and (for the deterministic stages) the same numeric content as the notebook reference run.

### Phase 8 — Polish, packaging, docs

1. Hook the `QFileSystemWatcher` into the status bus so tab icons update live.
2. Add menu items: File → Open block, File → Reload, File → Quit; Help → Keyboard shortcuts; Help → About.
3. Write a short README section in the main `README.md` mirroring the "Block Annotator GUI" section.
4. Ensure `pip install -e .` works in `eye_repo_win` / `eye_repo_linux` (no new heavy deps without explicit user approval).

**Self-check 8:** Run the full pytest suite and `scripts/validate_preprocessing_gui_load.py` one last time.

---

## 6. Self-check protocol (the agent's safety net)

Two artefacts grow as the agent works:

### 6.1 `scripts/validate_preprocessing_gui_load.py`

Mirrors `scripts/validate_block_annotator_load.py`. After each phase, append a section that instantiates the new tab's widgets headlessly (offscreen QPA: `QT_QPA_PLATFORM=offscreen`) on the sample block and asserts that key widgets exist, key signals connect, and the tab's status_signature() is computable. Exit code 0 on success.

Run before declaring any phase complete:
```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python scripts/validate_preprocessing_gui_load.py --experiment-path D:\sample_data_for_eye_repo --animal PV_106 --block 015
```

### 6.2 pytest suite

Add at minimum one `tests/test_preprocessing_gui_<stage>.py` per phase. Tests must:

- Run without a display (use `pytest-qt` if it is already a dep; otherwise just import `PyQt6` and use `QApplication.instance() or QApplication([])` in fixtures).
- Use a `conftest.py` fixture `sample_block` returning a `BlockSync` for `PV_106 / 2025_09_04 / block_015` (skip the test if the path is missing, so CI on machines without the sample data does not break).
- Verify both **GUI plumbing** (signals, widget creation) and **algorithmic equivalence** with the notebooks (call the same promoted helpers and compare to a small golden snapshot saved under `tests/golden/preprocessing_gui/`).

Run before declaring any phase complete:
```powershell
pytest tests/test_preprocessing_gui_*.py -q
```

### 6.3 Notebook re-run guard

After §4.3's refactor, re-run all five notebooks top-to-bottom (programmatically via `jupyter nbconvert --to notebook --execute`) on the sample block. They must succeed and produce the same files. Add a `scripts/check_notebooks_still_work.py` that does this and is invoked once at the end of Phase 0.

---

## 7. Human acceptance — `PREPROCESSING_GUI_ACCEPTANCE.md`

At the end of Phase 8, create `PREPROCESSING_GUI_ACCEPTANCE.md` in the repo root. The user will work through this checklist on the sample block to confirm the GUI is ready.

The agent must populate the checklist with **one numbered manual test per UI feature**, each in this format:

```markdown
- [ ] **A.1 — Block discovery.** Launch with no args. In the startup dialog, pick
      experiment path `D:\sample_data_for_eye_repo`, animal `PV_106`, block `015`.
      Click OK. **Expected:** main window opens; Sync tab is active; block info
      bar reads `PV_106 / 2025_09_04 / block_015`; all five tab icons are grey
      (none of the outputs exist yet on a fresh run).
```

Required sections (mirroring stage tabs):

- **A. Launch & block selection** (5 checks: with args, without args, env vars, switching blocks, multi-block selection)
- **B. Sync tab — preparation** (handle_eye_videos / arena / OE / brightness; both auto and manual paths for TTL and ROI)
- **C. Sync tab — simple sync + shift correction** (open Bokeh plot, type shifts, apply, preview applied, confirm `eye_left_simple_sync.csv` etc. written)
- **D. Sync tab — final merge + verify** (final_sync_df + sanity plot + verify stats >99% match)
- **E. Sync tab — DLC + jitter + finalize** (full chain ending with `left_eye_data.csv` / `right_eye_data.csv`)
- **F. Verify tab** (each EllipseVerifierWidget button works; Kerr ref picking saves to `self_kerr_refs.csv`)
- **G. Kerr tab** (calculate + export + downstream CSVs)
- **H. Behavior tab** (lizMov load, rolling window, threshold drag, export `block_015_behavior_state.csv`)
- **I. Sync-free tab** (each of the four sub-stages, staleness warning when applicable)
- **J. Multi-block batch** (run DLC + jitter on at least two blocks via the batch button)
- **K. Cross-check vs notebooks** (single test: pick a block, run the notebooks end-to-end on it in a copy folder; then run the GUI on the same block in a fresh copy; binary or column-wise compare `analysis/` folders; expected: identical for deterministic outputs, within ±1 frame on slider-corrected outputs)

Each item must have an explicit **Expected** clause so a third party can run the checklist without context.

---

## 8. Explicit non-goals (v1)

- `manual_outlier_annotation.ipynb` workflow.
- LFP / electrophysiology validation (`lfp_led_validation.ipynb`).
- Editing or annotating events (that is the Block Annotator's job).
- Writing a new synchronization algorithm or replacing `BlockSync`.
- Cross-platform GUI testing on Linux/macOS/Windows (`eye_repo_win` / `eye_repo_linux`; headless validation via offscreen QPA).
- New heavy dependencies. Allowed: packages already in `environment_win.yml` / `environment_linux.yml`.

---

## 9. Open items the agent must confirm with the user before starting

Even with the locked decisions in §1, the following still need a one-line user confirmation:

1. **`pyqtgraph` availability.** Included in both platform env files. Native plots in the GUI depend on it.
2. **`pytest-qt` availability.** Confirm it is acceptable to add `pytest-qt` for headless widget testing.
3. **Notebook re-execution as a self-check.** Confirm running notebooks via `jupyter nbconvert --execute` on the sample block as part of Phase 0's exit criterion is acceptable (it will write into the sample block's `analysis/` folder — the agent should work on a copy).
4. **Where the GUI's defaults live.** A `~/.pets_preproc_gui_config.yaml` for last-used experiment path / animal / batch selection, or per-output-folder like the Block Annotator's `annotator_config.yaml`? Pick one and confirm.
5. **Behavior tab without `lizMov.mat`.** The sample block's Stage 4 currently depends on a MATLAB-produced `lizMov.mat`. Confirm whether the GUI should also offer a path to compute it from Python (out of scope per §8 unless requested), or just show the instructional error.

When all five are confirmed, the agent may begin Phase 0.
