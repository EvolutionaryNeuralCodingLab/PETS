# Reverse follow-up after large saccades (head still vs moving)

Reviewer Q6 simplified. Filled response: `reviewer_response.md`.

The reviewer’s verbatim “back-and-forth a couple of times” sequence (large saccade plus ≥2 smaller extras with a direction reversal) is in `double_steps_rev_verbatim/`. Frequency **during head movement only** (no still comparison) is in `double_steps_rev_verbatim/move_only/`.

## Rule

- Unique gaze events (paper 34 ms L/R pairing) on blocks with `lizMov.mat`.
- Large: `net_angular_disp` ≥ 7° (cohort P75, rounded down to 0.5°).
- Reverse follow-up: another **detected** saccade, not the binocular partner, onset in (t0, t0+100 ms], circular direction difference > 90°.
- Head moving vs still: any `lizMov` sample overlapping the primary on–off span.

## Run

```bash
PYTHONPATH=src python -m eye_tracking_system_tools.analysis.movement_associated_nystagmus \
  --registry configs/paper_blocks.yaml \
  --params configs/analysis_params.yaml \
  --out development/movement_associated_nystagmus \
  --cache-meta development/Rev_3_Q6_nystagmus_like/metadata
```
