# Kerr relative error (fig 2b)

Quantifies how Kerr ellipse-to-angle bias survives after the paper’s rest-zero.
Lookup uses **unzeroed** `(k_phi, k_theta)`; comparison is in **relative** coordinates
(full-block nanmedian, same as fig 2b).

## Run

1. Lab volume mounted so `PV_126` `block_007` in [`configs/paper_blocks.yaml`](../../configs/paper_blocks.yaml) exists.
2. Kernel: `eye_repo_mac`. From the repo root, `PYTHONPATH=src` (the notebook also inserts `src` itself).
3. Open `kerr_relative_error.ipynb` and run all cells.

The catalog fig 2b window is 0–8 s. On the current verified CSVs that onset is all-NaN, so the notebook slides to the first 8 s with enough finite φ/θ (`pick_trace_window`).

Simulation table (copied next to this README):

`ellipse_angle_mapping_correct_diameter_08mm_distance_13mm.csv`

Eye diameter 8 mm, camera distance 13 mm. Ground-truth `(x_angle, y_angle)` vs Kerr `(phi, theta)`.

## Outputs

Written to `outputs/` beside this folder:

| File | Content |
|---|---|
| `kerr_error_heatmap.pdf` | ±60° GT grid, 1° error magnitude, 5° GT→Kerr vectors |
| `figure_2b_kerr_correction.pdf` | 8 s φ/θ overlay: original relative vs corrected-relative |
| `figure_2b_relative_residual.pdf` | Position offset between rest-centered traces (not travel) |
| `figure_2b_relative_residual.csv` | Position offset **and** path-length / step-size travel stats |
| `figure_2b_travel_path.pdf` | Cumulative inter-sample path, original vs corrected |
| `figure_2b_travel_steps.pdf` | Per-frame step scatter (unity line = no stretch) |
| `figure_2b_component_error_hist.pdf` | Full-block Δφ and Δθ residual histograms (mean ± SEM) |
| `figure_2b_component_error_2d.pdf` | Per-frame error vectors (Δφ, Δθ); mean arrow + 1σ ellipse |
| `figure_2b_component_error.csv` | Mean, std, SEM of Δφ and Δθ per eye |
| `cohort_block_readiness.csv` | Jitter-style QC: mounted? both `*degrees_raw_verified*` CSVs with φ/θ? |
| `cohort_component_error_per_block.csv` | Every ready paper block: signed and |Δ| stats |
| `cohort_component_error_per_animal.csv` | Same table (copy; one row per block × eye × axis) |
| `cohort_component_error_hist.pdf` | Per-animal Δφ / Δθ residual histograms (blocks pooled) |
| `cohort_component_error_across_animals.csv` | Animal-level mean of blocks, then across-animal mean/SD |
| `cohort_component_error_across_animals.pdf` | Animal bars (mean of blocks) with per-block dots |
