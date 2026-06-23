# Preprocessing GUI — agent handoff prompt

**Copy everything in the "Start here" section below into a new Cursor Agent chat** to resume implementation. Update this file when you finish a phase or discover new gaps.

---

## Start here (paste into new agent)

You are implementing the **PETS Preprocessing GUI** for this repository.

### Required reading (in order)

1. **`development/plans/PREPROCESSING_GUI_AGENT_PLAN.md`** — single source of truth (architecture, inventory, phases 0–8, acceptance gates).
2. **`development/PREPROCESSING_GUI_ACCEPTANCE.md`** — scope, gaps, sync-free artifact model.
3. **This handoff** — current progress and workflow rules.

### Mission (unchanged)

Build `eye_tracking_system_tools.annotation.preprocessing_gui` — PyQt6 tabbed app:

```powershell
python -m eye_tracking_system_tools.annotation.preprocessing_gui
```

**Environment:** `environment_win.yml` → `eye_repo_win` (Windows) or `environment_linux.yml` → `eye_repo_linux`.

**Reference blocks:** PV_106 block_015 (sync); PV_126 block_006 (behavior / lizMov).

### Current progress

| Phase | Scope | Status |
|-------|--------|--------|
| **0–7** | Full GUI + sync-free unified artifacts | **DONE** |
| **8** | Polish, unified env, README tutorial, acceptance doc | **DONE** |

#### Remaining gaps (non-blocking)

1. Sync tab in-tab `channeldict` / `bad_blocks` editor.
2. `test_find_jittery_frames_matches_notebook`.
3. Golden snapshots under `tests/golden/preprocessing_gui/`.
4. MATLAB `getLizMovement` still external for Behavior tab.

### Self-check commands

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
pytest tests/test_preprocessing_gui_*.py -q
python scripts/validate_preprocessing_gui_load.py `
  --experiment-path D:\sample_data_for_eye_repo --animal PV_106 --block 015
```

---

## Handoff changelog

| Date | Branch | Notes |
|------|--------|-------|
| 2026-06-20 | `preprocessing-gui-implementation` | Env cleanup: `eye_repo_win` + `eye_repo_linux`; deprecated annotator envs removed |
| 2026-06-20 | `preprocessing-gui-implementation` | Phase 7 Sync-free tab |
| 2026-06-20 | `preprocessing-gui-implementation` | Phase 6 Behavior tab |
