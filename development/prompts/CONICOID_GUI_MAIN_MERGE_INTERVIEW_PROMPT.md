# Conicoid GUI → main (clean merge) — Agent Interview Prompt

**Purpose:** Context prompt for a **new Cursor agent**. That agent must **interview the user first**, lock an implementation/merge plan, and only then help prepare a **narrow, tested** path into `main`. It must **not** open a huge PR into `main` or “fix forward” untested GUI changes onto `main`.

**Related savepoint (already on this branch, do not redo):**  
`physical_eye_simulation/` — compact Kerr vs Swirski simulation package + README + one reference heatmap. Ready to ride into `main` **after** the Conicoid/GUI work is approved. PR **#3 into `main` was closed** because it dragged the entire feature history; do not reopen that shape of PR.

---

## Workflow (mandatory order)

| Step | Agent role | Output |
|------|------------|--------|
| **1 — Interview** | Requirements agent (you) | Decision log + draft merge/implementation plan |
| **2 — Plan lock** | Same agent / user review | Approved plan file under `development/plans/` |
| **3 — Build & verify** | Builder agent (later or same after sign-off) | Fixes + GUI/algorithm changes **on a feature branch**, with tests the user can run |
| **4 — Merge to main** | Only after user sign-off | Small, reviewable PR(s) into `main` of **already-tested** modules |

**Do not write application code, reshape git history destructively, or open a PR to `main` until the user explicitly approves the plan.**  
Interview in **batches** (5–8 questions). Prefer multiple-choice + short free text. Record every answer in the Decision Log.

---

## User’s merge philosophy (non-negotiable)

1. **`main` stays clean.** Pull into `main` only modules the user has verified behave as intended.
2. **No re-work on `main` after merge.** Fix and finalize on the feature line first.
3. **Organize before merging.** Prefer splitting or sequencing PRs if one branch is too large (simulation savepoint vs Conicoid GUI vs ellipse-rotation, etc.)—**ask the user** which split they want; do not invent a force-push story without approval.
4. **`physical_eye_simulation/` is already the compact savepoint** for the simulation comparison. Treat it as done unless the interview finds a concrete gap.

---

## Current git context (read this first)

| Item | Detail |
|------|--------|
| Working branch | `feature/pogona_eye_simulation` (tracks `origin/feature/pogona_eye_simulation`) |
| Lineage | Cut from / includes history of `feature/conicoid_tab_kerr_check` and earlier GUI work — **not** a tiny tip commit vs `main` |
| Closed PR | [#3](https://github.com/EvolutionaryNeuralCodingLab/PETS/pull/3) into `main` — closed without merge (too large / premature) |
| Simulation savepoint commit | `9fceb07` — adds `physical_eye_simulation/`, tests, `.gitignore` notes |
| Base for eventual main PR | Almost certainly **`main`**, but **only** after Conicoid GUI is user-approved; may need an intermediate PR strategy (ask user) |

### What already exists on this branch (ground truth — verify by reading)

Summarize these in 5–10 bullets after reading, then interview:

| Area | Locations |
|------|-----------|
| Conicoid package | `src/eye_tracking_system_tools/preprocessing/conicoid/` |
| Writers / compare | `calculate_conicoid_angles.py`, `compare_conicoid_kerr.py` |
| GUI Conicoid tab | `src/.../preprocessing_gui/tabs/conicoid_tab.py` |
| GUI shell | `preprocessing_gui/app.py`, `batch_runner.py`, other tabs (Sync, Verify, Kerr, …) |
| Docs | `docs/preprocessing.md` (Conicoid + Kerr check); older notes in `docs/handoff_conicoid_kerr.md` (partly stale on branch names) |
| Scoring rules | 3D angle between normals; Kerr chart ≠ gaze; prefer `c_nx/c_ny/c_nz` |
| Simulation | `physical_eye_simulation/` — analytic default, optional Blender; Pogona geometry; Kerr residual vs Swirski ~0° analytic |
| Tests | `tests/test_preprocessing_gui_conicoid.py`, conicoid unit tests, `tests/test_eye_simulation_projection.py` |

**User report (high level, to be refined in interview):** some Conicoid GUI content is **too much**, some is **too shallow**; there are **bugs encountered while using the GUI**; algorithms need **finalizing** (defaults, RANSAC, focal length source, principal point, Dierkes vs Swirski, refine/refraction knobs, etc.).

---

## Product intent (not yet locked — interview to lock)

- **Swirski / conicoid** = trusted gaze for quantitative work (`c_nx/c_ny/c_nz`).
- **Kerr** = retained sanity check / legacy consumers (`k_phi` / `k_theta`); do not overwrite Kerr columns.
- Preprocessing GUI should let a **naive block** run Conicoid (and compare to Kerr when Kerr outputs exist) in a way that matches the user’s real workflow—not a kitchen-sink research panel.
- Ellipse-rotation / geometry v2 correctness must be settled before trusting live ellipses (confirm with user what is already shipped vs still buggy).
- Simulation package documents *why* Swirski won; GUI should make that path usable on real data.

---

## Interview agenda (run in batches)

### Batch A — Scope & merge shape

1. Target of the **next** PR: `main` directly vs intermediate branch (name?) vs PR into `feature/conicoid_tab_kerr_check` first?
2. Should **`physical_eye_simulation/`** be in the same PR as the GUI, or a **separate** PR that lands only after GUI sign-off?
3. Is history rewrite (rebase/split onto fresh branch from `main`) allowed, or only additive commits / stacked PRs?
4. What must **never** land on `main` from this line (e.g. development scratch, MRI paths, personal configs)?

### Batch B — Conicoid GUI: too much / too shallow

5. Walk the Conicoid tab: which controls/sections to **remove or hide** (refraction, Dierkes, refine, spin_max, batch rotation, …)?
6. Which missing pieces to **add or deepen** (defaults, progress, error messages, single “Run Conicoid” button, auto-compare with Kerr, focal-length readout, …)?
7. Tab order / naming: keep “Conicoid” or rename to “Swirski” / “Gaze (sphere)”?
8. Batch runner: Conicoid automatic after Kerr, or only on demand?

### Batch C — Bugs the user already hit

9. List each bug (symptom, block/path if any, repro steps, expected vs actual). Agent must catalogue these before proposing fixes.
10. Any regressions in Sync / Verify / Kerr / Explore when Conicoid is present?
11. Environment: which conda env / OS for the verification checklist?

### Batch D — Algorithm finalization

12. Sphere method default: `swirski_2d` only, or keep `dierkes_3d` as advanced?
13. RANSAC on for production? Any knobs exposed in GUI?
14. Focal length source priority: `LR_focal_length.csv` → ? → 652.6 px fallback — confirm.
15. Principal point: Kerr ref from Verify vs image centre — confirm.
16. Ellipse convention: geometry v2 only (`width`/`height` semi-axes, `phi` radians)? Any live blocks still on old convention?
17. Outputs: required columns/files; is `c_phi`/`c_theta` still written for old tools?
18. Compare-with-Kerr: always 3D angle; any plots/CSV names to freeze?

### Batch E — Acceptance & “merge-ready” bar

19. Minimum manual checklist on a real block before approving merge to `main`.
20. Required automated tests (existing + new).
21. Docs to update (`docs/preprocessing.md`, GUI help text, README).
22. Explicit sign-off: “Approve plan for builder agent?” / “Approve PR to `main`?” as separate gates.

---

## Hard rules for the interview agent

1. **Do not implement** until the plan is approved.
2. **Do not open or reopen a mega-PR into `main`.**
3. **Do not force-push** to `main` or shared branches unless the user explicitly requests it in writing in the interview.
4. Start by **reading** the files in the table above and summarizing current behavior.
5. Keep a **Decision Log** (below). Update it every batch.
6. Prefer **phased** delivery: (0) bugfix + algorithm defaults, (1) GUI slim/deepen, (2) tests + docs, (3) merge strategy into `main`, (4) optional separate simulation PR if split.
7. End the interview phase with: filled decision log, proposed PR split, acceptance checklist, and a clear **user sign-off question**.

---

## Decision log template

| # | Topic | Options considered | Decision | Date |
|---|--------|-------------------|----------|------|
| 1 | Next PR base / target | | | |
| 2 | Simulation package same PR or separate | | | |
| 3 | History rewrite allowed? | | | |
| 4 | Conicoid GUI remove list | | | |
| 5 | Conicoid GUI add/deepen list | | | |
| 6 | Bugs to fix (link to catalogue) | | | |
| 7 | Algorithm defaults locked | | | |
| 8 | Acceptance checklist | | | |
| 9 | Plan approved for builder? | Y/N | | |
| 10 | Approved for PR to main? | Y/N | | |

---

## Deliverables when interview completes

Create (after user approval of the outline):

`development/plans/CONICOID_GUI_MAIN_MERGE_PLAN.md`

containing:

- Locked decisions from the log  
- Bug catalogue with repro  
- GUI before/after scope  
- Algorithm defaults  
- Test + manual acceptance plan  
- Exact git/PR sequence so **`main` only receives tested modules**  
- Explicit note that `physical_eye_simulation/` is the simulation savepoint already on `feature/pogona_eye_simulation`

---

## Starter message for the user (agent may paste)

> I’ll help prepare a **clean** Conicoid/GUI path into `main` without another oversized PR. I’ll first inventory what’s on `feature/pogona_eye_simulation`, then interview you in short batches about GUI scope (too much / too shallow), bugs you’ve hit, algorithm defaults, and how you want PRs split. I won’t implement or open a PR to `main` until you approve the plan. The compact `physical_eye_simulation/` package is already saved on this branch for later.

---

## Pointers the agent should not ignore

- Closed PR commentary: simulation-only intent vs accidental full-branch PR.  
- `physical_eye_simulation/README.md` — Kerr residual vs Swirski; citations; analytic vs Blender.  
- `docs/preprocessing.md` — Conicoid as trusted gaze; Kerr as check; 3D disagreement.  
- Scoring: never subtract Kerr chart angles from command yaw/pitch; use normals + `angle_between_normals_deg`.
