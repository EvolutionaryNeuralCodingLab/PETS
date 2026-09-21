# Text-only statistical analyses (code, not S1_Data worksheets)

These results appear in manuscript prose, not in the editor-requested panel
list. They are **not** deposited as Excel worksheets. The publication task is
to keep the analysis code in the repository.

| Result | Code | Status |
|---|---|---|
| Head–saccade timing (median 78.1 ms; 59.4 / 36.3 / 4.3%) | `src/eye_tracking_system_tools/analysis/saccade_head_timing.py` | Present. Writes `saccade_head_offsets.csv` when run on lab blocks. |
| Corrective contralateral events (3.7% of 15,070 monocular) | `src/eye_tracking_system_tools/analysis/corrective_head.py` (`corrective_partners`, 80 ms / 45°) | Present. |
| Active-time fraction (10.2%) | `src/eye_tracking_system_tools/analysis/behavior_state.py` (epoch durations from annotated state CSVs) | Supporting code present. No dedicated “10.2%” exporter was found. |
| Rate vs state Monte Carlo (Δ = 2.66 Hz) | — | **Missing** as a named paper exporter. `monocular_species.py` / `saccade_robustness.py` implement a *different* shuffle (monocular-fraction species comparison). |
| Amplitude vs head-movement Monte Carlo (Δ = 0.64°) | — | **Missing** as a named paper exporter. |
| Pupil vs state Monte Carlo (p = 5.0e-05) | `figures_3e_3f_pupil.py` exports pupil samples; permutation test not found | **Missing** permutation script. |

Do not recompute or alter the published tests unless a separate manuscript
inconsistency is found.

Lab-block inputs (`behavior_state.csv`, `lizMov.mat`, eye CSVs) are not part
of the public git tree. Running the timing/corrective exporters requires the
processed experimental blocks on the lab mount.
