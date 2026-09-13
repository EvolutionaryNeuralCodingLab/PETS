# Back-and-forth sequences after large saccades (reviewer Q6, verbatim)

Filled response: `reviewer_response.md`.

This analysis matches the reviewer’s description of Video S1: after a large saccade the eye moves **back-and-forth a couple of times with small amplitude**. That is a **sequence**, not a single reverse follow-up (the parent folder).

Head-movement only (no still comparison; nystagmus frequency): `move_only/`.

## Rule

- Unique gaze events (paper 34 ms L/R pairing) on blocks with `lizMov.mat`.
- Large: `net_angular_disp` ≥ 7° (cohort P75, rounded down to 0.5°).
- Back-and-forth sequence: ≥2 additional **detected** saccades (not the binocular partner of the primary) with onset in (t0, t0+150 ms], each smaller than the primary when both amplitudes are finite, ≥1 counter-directed vs the primary (circular difference > 90°), and ≥1 consecutive pair of extras that reverse relative to each other.
- Head moving vs still: any `lizMov` sample overlapping the primary on–off span.

## Paper-cohort result

- Overall: **18.1%** (799/4416)
- Head moving: **21.3%** (639/2997)
- Head still: **11.3%** (160/1419)
- Fisher OR moving vs still = **2.13**, **P < 0.001**

## Run

```bash
PYTHONPATH=src python -m eye_tracking_system_tools.analysis.double_steps_rev_verbatim \
  --registry configs/paper_blocks.yaml \
  --params configs/analysis_params.yaml \
  --out development/movement_associated_nystagmus/double_steps_rev_verbatim \
  --cache-meta development/Rev_3_Q6_nystagmus_like/metadata
```
