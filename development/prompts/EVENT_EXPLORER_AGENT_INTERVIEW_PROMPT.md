# Event Explorer GUI — Agent Interview Prompt (Phase 1)

**Purpose:** This document is the **context prompt for a new Cursor Agent** whose job is **not** to build the GUI yet, but to **interview the user**, resolve ambiguities, and produce a locked **implementation plan** (`EVENT_EXPLORER_IMPLEMENTATION_PLAN.md`) for a separate builder agent.

---

## Workflow (three agents, in order)

| Step | Agent role | Input | Output |
|------|------------|-------|--------|
| **1 — Interview** | Requirements agent | **This file** + repo attachments | Completed decision log + draft plan sections |
| **2 — Plan lock** | Same or user review | Interview answers | `EVENT_EXPLORER_IMPLEMENTATION_PLAN.md` (approved) |
| **3 — Build** | Builder agent | Approved plan + codebase | `src/eye_tracking_system_tools/annotation/event_explorer/` + tests + docs |

**The interview agent must not write application code, scaffold packages, or start implementation until the user explicitly approves the plan.**

---

## Hard rules for the interview agent

1. **Do not implement** — no new Python modules, no pyproject changes, no GUI mockups in code.
2. **Do not assume** — if a requirement is unclear, ask. Prefer structured multiple-choice + free-text follow-ups.
3. **Start by reading** the attached repo files (list below) and summarizing what already exists in 5–10 bullets.
4. **Conduct the interview in batches** — group related questions (5–8 per turn); wait for answers before the next batch.
5. **Record every decision** in a running **Decision Log** table (template at end of this file).
6. **Flag blockers** — list anything that requires lab data paths, schema changes, or pipeline outputs not yet standardized.
7. **End with deliverables:**
   - Filled decision log
   - Proposed module layout (names only)
   - Phased implementation outline (0–N with acceptance checks)
   - Open risks / deferred v2 items
   - **Explicit user sign-off question:** “Approve this plan for the builder agent?”

---

## Product vision (user intent — not yet locked)

Build a **second-stage analysis GUI** (same **`eye_annotator`** conda env / PyQt6 stack as Block Annotator) for **exploring annotated events** across blocks:

- **No video playback** in v1 (data / plots only).
- **Interactive table** of all annotated events (possibly across many blocks / animals / sessions).
- **Toggle / filter** by event type, block, animal, date, quality flags, notes, etc.
- **Per-event data views:** pull time-aligned snippets from each block’s `analysis/` folder and OE recording for preview, averaging, and export.
- **Data streams to support** (minimum set — confirm with user):
  - Open Ephys: headstage, ADC, AUX (via existing `OERecording` / `oe_streams.py` patterns)
  - Left / right eye dataframes from pipeline (`left_eye_data.csv`, `right_eye_data.csv`, `le_df.csv`, `re_df.csv`, columns such as `ms_axis`, ellipse coords, Kerr degrees if present)
  - Possibly `final_sync_df.csv` slices for arena-linked columns
- **Plot modes** (confirm with user):
  - Single-trial waveform / scatter / heatmap for one selected event
  - Event-type averages (mean ± SEM across N events)
  - Overlay many trials (spaghetti or aligned to `timepoint_ms`)
  - Export: CSV, PNG, NPZ, or “analysis bundle” per selection

This tool complements **`block_annotator`** (mark events) — it consumes its outputs.

---

## What already exists in the repo (ground truth)

### Block Annotator (shipped on branch `manual_annotation_gui`)

| Item | Location / detail |
|------|-------------------|
| Launch | `python -m eye_tracking_system_tools.annotation.block_annotator` |
| Env | `environment_annotator.yml` → conda env `eye_annotator`; extras `[annotator]` in `pyproject.toml` |
| Annotation output | `{output_folder}/{animal}_{date}_block_{num}_annotations.json` |
| JSON schema | `schema_version: 1`; fields: `animal_call`, `experiment_date`, `block_num`, `block_path`, `sample_rate_hz`, `events[]` |
| Each event | `id`, `event_type`, `timepoint_ms`, `start_ms`, `end_ms`, `range_half_width_ms`, `row_index`, `arena_frame`, `l_eye_frame`, `r_eye_frame`, `note` |
| Config | `{output_folder}/annotator_config.yaml` — `event_types`, default ±100 ms window |
| Time base | `ms_axis = Arena_TTL / (sample_rate/1000)` — same clock as `OERecording.get_data` (see comment in `oe_streams.py`) |
| Block loader | `block_loader.py` + `BlockSync` + `load_final_sync_df` |
| OE access | `OERecording.py`, `oe_streams.py` (HS / ADC / AUX enumeration, decimated fetch) |
| Plan precedent | `BLOCK_ANNOTATOR_IMPLEMENTATION_PLAN.md` — use same phased + acceptance-check style for the new plan |

### Block folder layout (PETS convention)

```
{path_to_animal}/{animal_call}/{experiment_date}/block_{NNN}/
  analysis/
    final_sync_df.csv          # master timeline (required for sync)
    left_eye_data.csv          # pipeline eye table (L)
    right_eye_data.csv         # pipeline eye table (R)
    le_df.csv / re_df.csv      # alternate / legacy names
  oe_files/ ...                # Open Ephys recording
  eye_videos/ ...              # not needed for Event Explorer v1
  arena_videos/ ...            # not needed for Event Explorer v1
```

### Known data quirks (inform UX / loader design)

- Early rows in `final_sync_df` may have **invalid frame sentinels** (INT64_MIN) before valid sync — events may reference valid `ms_axis` windows later in the block.
- Eye CSV column names vary (`L_eye_frame` vs `eye_frame`); Block Annotator already normalizes via helpers.
- OE trace loading is **CPU-heavy** — Block Annotator hides OE panel by default; Event Explorer should discuss caching / lazy load / precomputed snippets.

---

## Annotation JSON example (schema v1)

```json
{
  "schema_version": 1,
  "animal_call": "PV_106",
  "experiment_date": "2025_09_04",
  "block_num": "015",
  "block_path": "D:/sample_data_for_eye_repo/PV_106/2025_09_04/block_015",
  "sample_rate_hz": 20000,
  "events": [
    {
      "id": "uuid",
      "event_type": "saccade",
      "timepoint_ms": 12345.6,
      "start_ms": 12245.6,
      "end_ms": 12445.6,
      "range_half_width_ms": 100,
      "row_index": 3543,
      "arena_frame": 1070,
      "l_eye_frame": 0,
      "r_eye_frame": null,
      "note": ""
    }
  ]
}
```

---

## Proposed package location (suggestion only — confirm in interview)

```
src/eye_tracking_system_tools/annotation/
  event_explorer/           # NEW — parallel to block_annotator/
    __main__.py
    app.py
    models.py               # EventRecord, StreamDefinition, PlotSpec
    catalog.py              # scan output_folder → unified event table
    block_data_loader.py    # resolve block_path → CSVs + oe_rec
    snippet_extractor.py    # ms windows → aligned arrays
    plot_panel.py           # pyqtgraph / matplotlib embed
    export_io.py
    persistence.py          # explorer session / filters (optional)
```

Reuse from `block_annotator` where possible (`models.AnnotationEvent`, `config_io`, `oe_streams`, `block_loader` patterns) — **do not duplicate** OE alignment logic.

---

## Interview questionnaire (agent must cover all sections)

Ask the user these topics. Skip none. Offer defaults only as options, not decisions.

### A — Scope & users

1. **Primary user:** solo analyst, lab-wide, publication figures, or QC only?
2. **Single session vs catalog:** one output folder per run, or aggregate annotations from many folders / animals?
3. **v1 non-goals:** confirm no video, no re-annotation, no editing source JSON in v1?
4. **Relationship to notebooks:** replace `manual_outlier_annotation.ipynb` / saccade pipelines eventually, or parallel forever?

### B — Inputs & discovery

5. **Annotation source:** only `*_annotations.json` from Block Annotator, or also import from other formats?
6. **Root picker at startup:** same `{output_folder}` as annotator, or a higher-level “study root” that scans recursively?
7. **Block path resolution:** trust `block_path` in JSON, or re-resolve from `animal_call` / `date` / `block_num` if folder moved?
8. **Missing data policy:** if `left_eye_data.csv` missing but `le_df.csv` exists — fallback order? Show warning in table?
9. **Stale sync guard:** honor preprocessing README rule (remap if `final_sync_df` newer than eye CSVs)?

### C — Event table UX

10. **Columns shown** in the master table (event_type, animal, block, timepoint_ms, note, frame IDs, …)?
11. **Filters:** by event type multi-select, block, date range, ms range, text search on notes?
12. **Sorting / grouping:** group by event_type for averages?
13. **Selection model:** single row, multi-select for average, “all of type X”?
14. **Row actions:** preview, export, mark exclude/reject, add tag for downstream?

### D — Time windows & alignment

15. **Default snippet window:** use each event’s `[start_ms, end_ms]`, or fixed ±X ms relative to `timepoint_ms`, or user-configurable global default?
16. **Alignment reference for averages:** align all trials to `timepoint_ms = 0`, or to `start_ms`, or stimulus-locked column if present?
17. **Eye data alignment:** slice eye CSVs on `ms_axis` or on `eye_frame` ranges from event metadata?
18. **OE alignment:** confirm `start_time_ms` = absolute ms in zeroed OE clock (same as Block Annotator) — any block-specific offset exceptions?

### E — Data streams & plots

19. **Mandatory streams for v1** — check all that apply:
    - OE headstage (which channels / user picks / all?)
    - OE ADC / AUX
    - L eye: which columns (center_x/y, Kerr, pupil area, …)?
    - R eye: same
    - `final_sync_df` columns (L_values, R_values, arena, …)
    - Accelerometer / movement derived files if present
20. **Plot types per stream:** time series, scatter (x vs y), image heatmap, PSD, rasters?
21. **Average plot:** mean ± SEM, median ± IQR, show N trials in legend?
22. **Single-trial inspection:** click row → update all stream panels, or pick one “primary” stream?
23. **Plotting backend:** pyqtgraph only (match annotator), matplotlib for export, or both?

### F — Performance & caching

24. **Expected scale:** events per session (10², 10³, 10⁴)? blocks per study?
25. **Precompute snippets** to disk on first load vs on-demand `get_data` per click?
26. **Cache location:** next to annotations, system temp, or `{output_folder}/snippet_cache/`?
27. **Background threads** for load/plot OK, or keep UI strictly single-threaded with progress bar?

### G — Export & reproducibility

28. **Export formats:** CSV per trial, combined long-format CSV, NPZ, HDF5, PDF report?
29. **Export naming convention** and folder structure?
30. **Provenance metadata** embedded in exports (block_path, event id, git hash, tool version)?
31. **Save explorer session** (filters, selected streams, plot settings) to YAML/JSON?

### H — Config & environment

32. **Same conda env** (`eye_annotator`) or new env file?
33. **CLI launch** like annotator (`--catalog`, `--output`) or dialog-only?
34. **Shared config** with `annotator_config.yaml` or separate `explorer_config.yaml`?

### I — Testing & acceptance

35. **Sample data path** for CI / manual QA (user’s `D:\sample_data_for_eye_repo\...`)?
36. **Definition of done for v1** — minimum demo script the user must run successfully?
37. **Phased delivery preference** (table-only → single-trial plots → averages → export)?

---

## Files to attach when starting the interview agent

| File | Why |
|------|-----|
| **This file** | Interview instructions |
| `BLOCK_ANNOTATOR_IMPLEMENTATION_PLAN.md` | Plan format precedent |
| `src/eye_tracking_system_tools/annotation/block_annotator/persistence.py` | JSON schema |
| `src/eye_tracking_system_tools/annotation/block_annotator/models.py` | Event model |
| `src/eye_tracking_system_tools/annotation/block_annotator/block_loader.py` | Block discovery |
| `src/eye_tracking_system_tools/annotation/block_annotator/oe_streams.py` | OE time alignment |
| `src/eye_tracking_system_tools/preprocessing/OERecording.py` | `get_data` API |
| `src/eye_tracking_system_tools/preprocessing/block_sync_core.py` | `load_final_sync_df` |
| `src/eye_tracking_system_tools/preprocessing/README.md` | Eye CSV conventions |
| `environment_annotator.yml` | Target environment |
| Example `*_annotations.json` from user’s output folder (if available) | Real event list |

---

## Decision log template (agent fills during interview)

| ID | Topic | Question | User answer | Status |
|----|-------|----------|-------------|--------|
| A1 | Scope | … | | pending |
| B1 | Inputs | … | | pending |
| … | | | | |

Status: `pending` → `decided` → `deferred_v2`

---

## Output format for `EVENT_EXPLORER_IMPLEMENTATION_PLAN.md` (after interview)

The interview agent should produce a plan mirroring `BLOCK_ANNOTATOR_IMPLEMENTATION_PLAN.md`:

1. **Product summary** (locked table)
2. **User requirements** (numbered, no ambiguity)
3. **Data model** — catalog row schema, snippet schema, export schema
4. **Architecture** — module tree, data flow diagram (ASCII or mermaid)
5. **Reuse map** — what to import from `block_annotator` / preprocessing
6. **Phases 0–N** with acceptance checks (agent must not proceed until checks pass)
7. **Known risks & mitigations**
8. **Agent prompt for builder** (copy-paste block for Phase 3)
9. **Approval record**

---

## Copy-paste prompt for the interview agent (start here)

```
You are the Event Explorer requirements agent for the PETS repo.

Read EVENT_EXPLORER_AGENT_INTERVIEW_PROMPT.md completely before doing anything else.

Your job is to INTERVIEW the user and produce EVENT_EXPLORER_IMPLEMENTATION_PLAN.md.
You must NOT write application code or modify pyproject.toml until the user explicitly approves the final plan.

Steps:
1. Read the attached repo files listed in the prompt.
2. Summarize existing Block Annotator outputs and block/analysis folder layout (brief).
3. Run the interview in batches using the questionnaire (sections A–I). Wait for answers.
4. Maintain the Decision Log table; resolve all v1 blockers.
5. Draft EVENT_EXPLORER_IMPLEMENTATION_PLAN.md with phased acceptance checks.
6. Ask: "Approve this plan for the builder agent?"

Do not start implementation. Do not skip clarification questions even if you think you know the answer from context.
```

---

## Copy-paste prompt for the builder agent (use only after plan approval)

```
You are implementing the PETS Event Explorer GUI. Read and follow EVENT_EXPLORER_IMPLEMENTATION_PLAN.md in the repo root completely.

Constraints:
- PyQt6 + pyqtgraph (same eye_annotator env as Block Annotator); no video playback in v1.
- Consume Block Annotator JSON outputs; load block analysis CSVs + OE via existing preprocessing modules.
- Interactive event table; single-trial and averaged plots; export as specified in the plan.
- New module under src/eye_tracking_system_tools/annotation/event_explorer/
- Reuse block_annotator / OERecording patterns; do not duplicate ms_axis ↔ OE alignment logic.
- Work in phases from the plan; run acceptance checks after each phase before continuing.

Deliverables: working module, tests, README section, no unrelated refactors.
```

---

## Approval record

| Date | Milestone |
|------|-----------|
| 2026-05-22 | User requested interview-first workflow; context prompt authored alongside Block Annotator on `manual_annotation_gui`. |
| | Interview completed: _pending_ |
| | Plan approved: _pending_ |
| | Builder agent started: _pending_ |
