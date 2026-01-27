# Figure 2i: Polar distributions of saccade directions

## Description

Polar distributions of saccade directions per eye and animal, rotated to align the eye's dominant axis of motion to the 0/180 degrees axis. The figure shows circular histograms with a baseline probability reference (light gray circle = 0.005 density).

## Data File Required

Place the `figure_2i.pickle` file in this directory. This file should be exported using the `export_figure_2i_data` function from `export_figures_2f_2h_2i_helper.py` in the `multiple_figures_pipeline_migration.ipynb` notebook.

## Expected Data Structure

The pickle file should contain a dictionary with the following keys:

- `figure_name`: String identifier ("figure_2i")
- `per_eye_histograms`: Dictionary keyed by animal ID, each containing:
  - `L`: Dictionary with left eye histogram data:
    - `angles`: np.ndarray of original saccade angles (degrees), float32
    - `counts`: np.ndarray of density per bin (original histogram), float32
    - `centers`: np.ndarray of bin centers for original histogram (degrees), float32
    - `rotated_angles`: np.ndarray of rotated angles (degrees, aligned to dominant axis), float32
    - `rotated_counts`: np.ndarray of density per bin (rotated histogram), float32
    - `rotated_centers`: np.ndarray of bin centers for rotated histogram (degrees), float32
    - `dominant_axis`: float, dominant axis orientation in degrees [0, 180)
    - `rotation_angle`: float, rotation angle applied in degrees [-90, 90]
    - `n_saccades`: int, number of saccades for this eye/animal
  - `R`: Dictionary with right eye histogram data (same structure as `L`)
- `rotation_catalog`: Dictionary mapping (animal, eye) tuples to rotation angle in degrees
  - Format: `{(animal_id, 'L'): rotation_angle, (animal_id, 'R'): rotation_angle, ...}`
- `num_bins`: Integer number of bins for the polar histogram (typically 36 for 10-degree bins)
- `baseline_prob`: Float baseline probability for the reference circle (typically 0.005)

## Usage

```bash
python figure_2i.py
```

The script will generate `figure_2i.pdf` in this directory.

## Notes

- Angles are in degrees and should be converted to radians for polar plotting
- The rotation aligns each eye's dominant axis to 0/180 degrees
- The baseline probability circle provides a visual reference for uniform distribution
- Histograms are typically normalized to probability distributions