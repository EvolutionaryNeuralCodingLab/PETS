# Figure 1e: Camera Jitter Quantification

## Description
Histogram of camera movements, quantified as percentage of overall frames in each bin, indicating low camera jitter.

## Data File
- `distances.pkl`: NumPy array containing displacement distances in micrometers (454,300 measurements)

## Reproduction
Run the reproduction script:
```bash
python src/eye_tracking_system_tools/figures/reproduction/main_figures/figure_1e.py
```

This will generate `figure_1e.pdf` in this directory.

## Figure Specifications
- **Type**: Histogram
- **Bins**: 15 bins from 0 to 500 μm
- **Y-axis**: Percentage of frames (%)
- **X-axis**: Displacement (μm)
- **Figure size**: 2 × 1.6 inches
- **DPI**: 150

## Original Pipeline
Created using `deprecated_pipelines_for_reference/figures_creation_jitter_quantification.ipynb`
