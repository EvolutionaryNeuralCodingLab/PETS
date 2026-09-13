# Reviewer Q6 — follow-up eye movements after large still-head saccades

Quantitative answer to the video-S1 comment about small back-and-forth eye
movements after a saccade. Outputs are written in place:

```
development/Rev_3_Q6_nystagmus_like/
  plots/          PDFs + PNGs
  metadata/       CSV / YAML / pickle / LOGIC.md
  captions.md     figure captions + numeric summary
```

The analysis **reuses the existing saccade table and `lizMov.mat` bout onsets**.
It does not re-detect saccades and does not label follow-ups as nystagmus, VOR,
or corrective saccades.

## How to run

From the repo root, kernel / env `eye_repo_mac`:

```bash
PYTHONPATH=src python -m eye_tracking_system_tools.analysis.secondary_after_saccade \
  --registry configs/paper_blocks.yaml \
  --params configs/analysis_params.yaml \
  --out development/Rev_3_Q6_nystagmus_like
```

## Selection (locked in the exporter)

1. Unique gaze events: concurrent L/R onsets within `binocular.sync_diff_ms`
   (34 ms) collapse to one event. Amplitude is `max(net_angular_disp)` of the
   paired eyes; direction is that of the larger eye.
2. Head still at onset: no `lizMov` movement sample in `[t_on − 50 ms, t_on)`.
   This is stricter than the during-saccade `head_movement` flag, so a delayed
   head bout that starts *during* the saccade is still allowed.
3. Large: 75th percentile of those still-head amplitudes, rounded down to 0.5°,
   not below 5° (one Fig 2e / main-sequence bin). The histogram figure shows
   median, P75, and the threshold.
4. Follow-ups: existing detected events with onset in `(t_on, t_on + 250 ms]`,
   excluding the L/R constituents of the primary event.
5. Reverse vs same: circular difference of `overall_angle_deg` > 90° vs ≤ 90°.
6. Subsequent head: next `lizMov` bout onset in the same 250 ms window.
7. Final position: primary-eye displacement along the primary movement axis at
   `t_on + 250 ms`, divided by primary amplitude (1 = stayed put, 0 = returned).

See `captions.md` and `metadata/summary.yaml` after the run.

## Results (paper lizard cohort)

Blocks with `lizMov.mat` only (PV_57 dropped). Unique still-head events with finite amplitude: **7006**. Large-event threshold: **5.5°** (P75 = 5.84°, rounded down; floor 5°). Qualifying events: **1903**.

| Quantity | Value |
|---|---|
| Any subsequent detected eye movement in 250 ms | **59.0%** (1123/1903) |
| First follow-up reverse / same / none | **35.9% / 20.8% / 41.0%** (2.3% unclassified direction) |
| ≥2 follow-ups; both directions in the window | **37.5%; 21.7%** |
| Median first-follow-up latency; amplitude | **68 ms; 2.3°** (primary median 9.6°) |
| Subsequent head-bout onset in the same window | **16.4%** (similar across follow-up classes: 15–19%) |
| Median remaining fraction of primary displacement | **0.92** (84% stay ≥ 0.5). Reverse-class median **0.69**; no-follow-up **0.99**. |

The eye typically remains at a displaced post-saccadic position. Follow-ups are usually smaller than the primary and are not strongly associated with later head-bout onset. Interpretation (nystagmus / VOR / “corrective”) is left open.
