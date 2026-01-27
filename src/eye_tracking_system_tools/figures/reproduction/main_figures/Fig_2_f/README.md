# Figure 2f: Inter-ocular peak-speed coupling

## Description

Two-dimensional probability histogram of right vs left-eye peak angular speed [°/ms] for all saccades during head-stationary epochs pooled across animals (equal-animal weighting). The figure consists of two panels: a macro view showing the full range and a micro view showing the low-speed region.

## Data File Required

Place the `figure_2f.pickle` file in this directory. This file should be exported using the `export_figure_2f_data` function from `export_figures_2f_2h_2i_helper.py` in the `multiple_figures_pipeline_migration.ipynb` notebook.

## Expected Data Structure

The pickle file should contain a dictionary with the following keys:

- `figure_name`: String identifier ("figure_2f")
- `right_eye_speeds`: np.ndarray of right eye peak speeds (deg/ms), float32
- `left_eye_speeds`: np.ndarray of left eye peak speeds (deg/ms), float32
- `weights`: Optional np.ndarray of weights for equal-animal weighting, float32
- `bins`: Integer number of bins for the histogram
- `macro_range`: Tuple (min, max) for the macro panel range
- `micro_range`: Tuple (min, max) for the micro panel range
- `macro_n_ticks`: Integer number of ticks for macro panel
- `micro_n_ticks`: Integer number of ticks for micro panel
- `macro_tick_list`: Optional list of custom tick positions for macro panel
- `micro_tick_list`: Optional list of custom tick positions for micro panel
- `iqr_multiplier`: Float IQR multiplier used for filtering
- `macro`: Dictionary with pre-computed macro histogram data:
  - `xedges`: np.ndarray of x bin edges, float32
  - `yedges`: np.ndarray of y bin edges, float32
  - `norm_counts`: np.ndarray of normalized counts (probability), float32
- `micro`: Dictionary with pre-computed micro histogram data:
  - `xedges`: np.ndarray of x bin edges, float32
  - `yedges`: np.ndarray of y bin edges, float32
  - `norm_counts`: np.ndarray of normalized counts (probability), float32
- `vmax_all`: Float maximum value across both histograms for colorbar normalization

## Usage

```bash
python figure_2f.py
```

The script will generate `figure_2f.pdf` in this directory.

## Notes

- The histogram data is pre-computed and normalized to probabilities (sums to 1.0)
- Both macro and micro panels share the same colorbar scale (`vmax_all`)
- Data is filtered using IQR-based outlier removal before histogram computation