# Preprocessing GUI — agent handoff prompt

**Copy everything in the "Start here" section below into a new Cursor Agent chat** to resume implementation. Update this file when you finish a phase or discover new gaps.

---

## Start here (paste into new agent)

You are implementing the **PETS Preprocessing GUI** for this repository.

### Required reading (in order)

1. **`development/plans/PREPROCESSING_GUI_AGENT_PLAN.md`** — single source of truth (architecture, inventory, phases 0–8, acceptance gates). Follow phases **in order**; do not skip self-checks.
2. **`development/README.md`** — where agent-facing files live.
3. **This handoff** — current progress, gaps, and workflow rules.

### Mission (unchanged)

Build `eye_tracking_system_tools.annotation.preprocessing_gui` — a PyQt6 tabbed app launched via:

```powershell
python -m eye_tracking_system_tools.annotation.preprocessing_gui
```

It wraps the five preprocessing notebooks (sync, verify, Kerr, behavior, sync-free) by calling existing `BlockSync` / `notebook_helpers` code verbatim. `manual_outlier_annotation.ipynb` is out of scope for v1.

**Environment:** `eye_annotator` conda env on Windows. Use `cv2` headlessly only (no `cv2.imshow`). Native plots use **pyqtgraph** (already in `environment_annotator_windows.yml`).

**Reference sample block:** `D:\sample_data_for_eye_repo\PV_106\2025_09_04\block_015`

### Current progress (as of branch `preprocessing-gui-implementation`)

| Phase | Scope | Status |
|-------|--------|--------|
| **0** | Skeleton, `notebook_helpers.py`, notebooks refactored, headless validation | **DONE** |
| **1** | Sync tab deterministic steps 1.2–1.12 (minus manual fallbacks) | **DONE** |
| **2** | Sync tab DLC + jitter + batch (1.13–1.18) | **DONE** |
| **3** | Manual TTL dialog + Qt ROI picker | **NOT STARTED** ← **you start here** |
| **4** | Verify tab + `EllipseVerifierWidget` | Not started |
| **5** | Kerr tab | Not started |
| **6** | Behavior tab (rolling window + threshold plot) | Stub only (`lizMov.mat` gate) |
| **7** | Sync-free tab | Not started |
| **8** | Polish (`QFileSystemWatcher`, README, `PREPROCESSING_GUI_ACCEPTANCE.md`) | Partial (menus only) |

#### What already exists in code

- Package: `src/eye_tracking_system_tools/annotation/preprocessing_gui/` (app, tabs, workers, batch_runner, bokeh_launcher, pyqtgraph_helpers, status_bus, config_io, block_picker)
- **Sync tab** (`tabs/sync_tab.py`): full workflow through `left_eye_data.csv` / `right_eye_data.csv` export
- Promoted helpers: `src/eye_tracking_system_tools/preprocessing/notebook_helpers.py`
- Tests: `tests/test_preprocessing_gui_phase0.py`, `tests/test_preprocessing_gui_sync.py`, `tests/conftest.py`
- Validation: `scripts/validate_preprocessing_gui_load.py`, `scripts/check_notebooks_still_work.py`

#### Missing modules (planned in §4.1 of the plan)

- `manual_ttl_dialog.py`
- `qt_roi_picker.py`
- `ellipse_verifier.py`

#### Known gaps vs plan (fix or track as you go)

1. **Phase 3 not built** — brightness uses auto-ROI only; OE parse has no manual TTL override UI.
2. **Sync tab 1.1** — no in-tab `channeldict` / `bad_blocks` editor (fields exist in `PreprocConfig` / `BlockHandle` only).
3. **Missing test** — `test_find_jittery_frames_matches_notebook` (Phase 2 self-check).
4. **No golden snapshots** — `tests/golden/preprocessing_gui/` not created.
5. **Downstream tabs** — Verify, Kerr, Behavior (beyond disable banner), Sync-free are placeholders.
6. **Status bus** — polling only; no `QFileSystemWatcher` (Phase 8).
7. **Acceptance doc** — `PREPROCESSING_GUI_ACCEPTANCE.md` not written (Phase 8).

#### Plan §9 open items — already decided

- `pyqtgraph` + `pytest-qt`: in annotator env files
- Config: per-output-folder `preproc_gui_config.yaml`
- No `lizMov.mat`: Behavior tab disabled with instructional message
- Notebook re-run guard: `scripts/check_notebooks_still_work.py` (uses `_phase0_sample_copy/`, gitignored)

### Your workflow rules

1. **One phase feature at a time** — complete Phase N self-check before Phase N+1.
2. **After each feature you build**, end your turn with a **Manual validation** subsection for the user:
   - Exact commands to run (pytest, validate script, GUI launch args)
   - Step-by-step UI clicks when human judgement is required
   - **Expected** outcome (files created, labels, tab icons, error messages)
   - Prefer **self-contained loops** (unit test + headless validate) when they cover the behavior; add manual GUI steps only when necessary (e.g. Bokeh slider, ellipse click-pick).
3. **Do not commit** unless the user asks. **Do** run tests locally before declaring a step done.
4. **Minimize scope** — reuse `BlockSync`, `notebook_helpers`, `block_annotator/video_widget.py` patterns; do not reimplement sync math.
5. **Update this handoff** when you complete a phase (status table + "next phase").

### Immediate next task: Phase 3

Implement per **§5 Phase 3** and **§3 rows 1.4–1.5** of the plan:

1. **`manual_ttl_dialog.py`** — Qt dialog producing `manual_line_map` + `arena_window`; reuse `_summarize_ttl_lines_from_events_csv` / `_plot_ttl_raster` from `BlockSync_class.py` (embed pyqtgraph or browser fallback).
2. **`qt_roi_picker.py`** — rubber-band ROI on first video frame; return `(x, y, w, h)` for `produce_frame_val_list_with_roi`.
3. Wire into **Sync tab**: show on auto failure or via "Override" buttons; no `input()` prompts.
4. Add pytest: `test_manual_ttl_dialog_payload_shape`, `test_qt_roi_picker_returns_tuple`.
5. Extend `scripts/validate_preprocessing_gui_load.py` if new widgets need smoke coverage.

### Self-check commands (run after every phase)

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
pytest tests/test_preprocessing_gui_*.py -q
python scripts/validate_preprocessing_gui_load.py `
  --experiment-path D:\sample_data_for_eye_repo `
  --animal PV_106 --block 015
```

### Key precedents to study before coding

- `src/eye_tracking_system_tools/annotation/block_annotator/` — app layout, video widget, config
- `scripts/validate_block_annotator_load.py` — headless load pattern
- `src/eye_tracking_system_tools/preprocessing/data_verification_utils.py` — ellipse corrector semantics (Phase 4)

---

## Handoff changelog

| Date | Branch | Notes |
|------|--------|-------|
| 2026-06-19 | `preprocessing-gui-implementation` | Phases 0–2 complete on Sync tab; dev docs moved to `development/`; resume at Phase 3 |
