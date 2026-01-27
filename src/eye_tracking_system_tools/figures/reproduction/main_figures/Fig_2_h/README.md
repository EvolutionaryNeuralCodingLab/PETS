# Figure 2h: Heatmaps of eye-endpoint densities

## Description

Heatmaps of eye-endpoint densities for concurrent and monocular saccades (single animal) relative to saccade start location. The figure compares synced (concurrent) vs non-synced (monocular) saccade trajectories.

## Data File Required

Place the `figure_2h.pickle` file in this directory. This file should be exported using the `export_figure_2h_data` function from `export_figures_2f_2h_2i_helper.py` in the `multiple_figures_pipeline_migration.ipynb` notebook.

## Expected Data Structure

The pickle file should contain a dictionary with the following keys:

- `figure_name`: String identifier ("figure_2h")
- `synced`: Dictionary with synced (concurrent) saccade trajectories:
  - `R`: Dictionary with right eye data:
    - `x`: np.ndarray of x-coordinates (angular displacement), float32
    - `y`: np.ndarray of y-coordinates (angular displacement), float32
  - `L`: Dictionary with left eye data:
    - `x`: np.ndarray of x-coordinates (angular displacement), float32
    - `y`: np.ndarray of y-coordinates (angular displacement), float32
- `non_synced`: Dictionary with non-synced (monocular) saccade trajectories:
  - `R`: Dictionary with right eye data:
    - `x`: np.ndarray of x-coordinates (angular displacement), float32
    - `y`: np.ndarray of y-coordinates (angular displacement), float32
  - `L`: Dictionary with left eye data:
    - `x`: np.ndarray of x-coordinates (angular displacement), float32
    - `y`: np.ndarray of y-coordinates (angular displacement), float32
- `rotation_catalog`: Optional dictionary mapping (animal, eye) tuples to rotation angle in degrees (for alignment)
- `nbins`: Integer number of bins for the heatmap
- `global_min`: Float minimum value for the heatmap extent
- `global_max`: Float maximum value for the heatmap extent

## Usage

```bash
python figure_2h.py
```

The script will generate `figure_2h.pdf` in this directory.

## Notes

- The x and y arrays represent angular displacements (zeroed to saccade start location)
- Trajectories are pooled across saccades to create density heatmaps
- The `rotation_catalog` may be used to align trajectories if needed
- Both synced and non-synced trajectories use the same extent (`global_min`, `global_max`)