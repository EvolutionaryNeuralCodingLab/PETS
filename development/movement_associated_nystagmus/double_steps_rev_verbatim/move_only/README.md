# Back-and-forth sequences during head movement (reviewer Q6)

Filled response: `reviewer_response.md`.

This is the head-movement-only version of the parent `double_steps_rev_verbatim` analysis. Head-stationary unique events are dropped before applying the large-saccade cutoff. There is no still-vs-moving comparison: the question is how often the Video S1 double-step occurs during head movement, interpreted as nystagmus-like gaze stabilization.

## Rule

- Unique gaze events (paper 34 ms L/R pairing) on blocks with `lizMov.mat`.
- Keep only events whose primary on–off span overlaps a `lizMov` sample.
- Large: `net_angular_disp` ≥ **7°**.
- Back-and-forth sequence: same as the parent folder (≥2 smaller extras in 150 ms, ≥1 reverse vs primary, ≥1 consecutive extra-to-extra reversal).

## Paper-cohort result

- Head-moving unique events: **10193** (7703 still dropped)
- Large during head movement: **2997**
- Back-and-forth sequence: **21.3%** (639/2997)

## Run

```bash
PYTHONPATH=src python -m eye_tracking_system_tools.analysis.double_steps_move_only \
  --registry configs/paper_blocks.yaml \
  --params configs/analysis_params.yaml \
  --out development/movement_associated_nystagmus/double_steps_rev_verbatim/move_only \
  --cache-meta development/Rev_3_Q6_nystagmus_like/metadata
```
